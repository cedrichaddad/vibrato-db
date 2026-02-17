use std::f32::consts::TAU;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::sleep;

const DIM: usize = 128;
const TOTAL_VECTORS: usize = 10_000;
const QUERY_START: usize = 5_000;
const QUERY_LEN: usize = 50;
const HTTP_RETRY_ATTEMPTS: usize = 6;

fn global_test_semaphore() -> Arc<Semaphore> {
    static SEM: OnceLock<Arc<Semaphore>> = OnceLock::new();
    SEM.get_or_init(|| Arc::new(Semaphore::new(1))).clone()
}

async fn acquire_test_lock() -> OwnedSemaphorePermit {
    global_test_semaphore()
        .acquire_owned()
        .await
        .expect("test semaphore closed")
}

fn retry_delay(attempt: usize) -> Duration {
    let exp = 1u64 << attempt.min(5);
    Duration::from_millis(25 * exp)
}

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, port: u16) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v2")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    for _ in 0..180 {
        if let Ok(resp) = client.get(&ready_url).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("server did not become ready at {}", ready_url))
}

async fn stop_server(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("identify-protocol")
        .arg("--roles")
        .arg("admin,query,ingest")
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "key-create failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(token) = line.strip_prefix("token=") {
            return Ok(token.trim().to_string());
        }
    }
    anyhow::bail!("token not found in key-create output: {}", stdout)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for lane in &mut vector {
        *lane /= norm;
    }
    vector
}

fn deterministic_frame(idx: usize) -> Vec<f32> {
    let mut state = splitmix64((idx as u64) ^ 0xD1B54A32D192ED03);
    let mut vector = Vec::with_capacity(DIM);
    for lane in 0..DIM {
        state = splitmix64(state ^ lane as u64);
        let unit = (state as f64 / u64::MAX as f64) as f32;
        vector.push(unit * 2.0 - 1.0);
    }
    normalize(vector)
}

fn deterministic_frame_for_track(track_seed: u64, idx: usize) -> Vec<f32> {
    let mut state =
        splitmix64(track_seed ^ (idx as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0x94D049BB133111EB);
    let mut vector = Vec::with_capacity(DIM);
    for lane in 0..DIM {
        state = splitmix64(state ^ ((lane as u64).wrapping_mul(0xBF58476D1CE4E5B9)));
        let unit = (state as f64 / u64::MAX as f64) as f32;
        vector.push(unit * 2.0 - 1.0);
    }
    normalize(vector)
}

fn gaussian_noise(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().clamp(1e-7, 1.0 - 1e-7);
    let u2 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = TAU * u2;
    r * theta.cos()
}

fn add_gaussian_noise(base: &[f32], sigma: f32, rng: &mut StdRng) -> Vec<f32> {
    let noisy = base
        .iter()
        .map(|value| *value + sigma * gaussian_noise(rng))
        .collect::<Vec<_>>();
    normalize(noisy)
}

async fn ingest_frame(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    idx: usize,
    vector: &[f32],
) -> usize {
    let body = serde_json::json!({
        "vector": vector,
        "metadata": {
            "source_file": "known_track.wav",
            "start_time_ms": idx * 20,
            "duration_ms": 20,
            "bpm": 128.0,
            "tags": ["identify", "protocol"]
        },
        "idempotency_key": format!("known-track-{idx}")
    });
    let mut last_err: Option<String> = None;
    for attempt in 0..HTTP_RETRY_ATTEMPTS {
        let req = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(token)
            .json(&body);
        match req.send().await {
            Ok(resp) => {
                if resp.status().is_server_error() && attempt + 1 < HTTP_RETRY_ATTEMPTS {
                    last_err = Some(format!("server status {}", resp.status()));
                    sleep(retry_delay(attempt)).await;
                    continue;
                }
                assert_eq!(resp.status(), StatusCode::CREATED);
                let payload: serde_json::Value = resp.json().await.expect("ingest json");
                return payload["data"]["id"].as_u64().expect("ingest id") as usize;
            }
            Err(err) => {
                if attempt + 1 == HTTP_RETRY_ATTEMPTS {
                    panic!("ingest request failed after retries: {err}");
                }
                last_err = Some(err.to_string());
                sleep(retry_delay(attempt)).await;
            }
        }
    }
    panic!(
        "ingest request exhausted retries: {}",
        last_err.unwrap_or_else(|| "unknown".to_string())
    );
}

async fn ingest_frame_for_track(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    global_idx: usize,
    local_idx: usize,
    vector: &[f32],
    source_file: &str,
) -> usize {
    let body = serde_json::json!({
        "vector": vector,
        "metadata": {
            "source_file": source_file,
            "start_time_ms": local_idx * 20,
            "duration_ms": 20,
            "bpm": 128.0,
            "tags": ["identify", "protocol", source_file]
        },
        "idempotency_key": format!("{source_file}-{global_idx}")
    });
    let mut last_err: Option<String> = None;
    for attempt in 0..HTTP_RETRY_ATTEMPTS {
        let req = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(token)
            .json(&body);
        match req.send().await {
            Ok(resp) => {
                if resp.status().is_server_error() && attempt + 1 < HTTP_RETRY_ATTEMPTS {
                    last_err = Some(format!("server status {}", resp.status()));
                    sleep(retry_delay(attempt)).await;
                    continue;
                }
                assert_eq!(resp.status(), StatusCode::CREATED);
                let payload: serde_json::Value = resp.json().await.expect("ingest track json");
                return payload["data"]["id"].as_u64().expect("ingest track id") as usize;
            }
            Err(err) => {
                if attempt + 1 == HTTP_RETRY_ATTEMPTS {
                    panic!("ingest track request failed after retries: {err}");
                }
                last_err = Some(err.to_string());
                sleep(retry_delay(attempt)).await;
            }
        }
    }
    panic!(
        "ingest track request exhausted retries: {}",
        last_err.unwrap_or_else(|| "unknown".to_string())
    );
}

async fn identify(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    vectors: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> serde_json::Value {
    let body = serde_json::json!({
        "vectors": vectors,
        "k": k,
        "ef": ef,
        "include_metadata": true
    });
    for attempt in 0..HTTP_RETRY_ATTEMPTS {
        let req = client
            .post(format!("{}/v2/identify", base_url))
            .bearer_auth(token)
            .json(&body);
        match req.send().await {
            Ok(resp) => {
                if resp.status().is_server_error() && attempt + 1 < HTTP_RETRY_ATTEMPTS {
                    sleep(retry_delay(attempt)).await;
                    continue;
                }
                assert_eq!(resp.status(), StatusCode::OK);
                return resp.json().await.expect("identify payload");
            }
            Err(err) => {
                if attempt + 1 == HTTP_RETRY_ATTEMPTS {
                    panic!("identify request failed after retries: {err}");
                }
                sleep(retry_delay(attempt)).await;
            }
        }
    }
    panic!("identify request exhausted retries");
}

#[tokio::test]
async fn identify_protocol_perfect_noisy_and_silent_anchor() {
    let _permit = acquire_test_lock().await;
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("identify_protocol_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping identify protocol test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping identify protocol test: failed to spawn server: {}",
                e
            );
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready");

    let dataset: Vec<Vec<f32>> = (0..TOTAL_VECTORS).map(deterministic_frame).collect();
    for (idx, vector) in dataset.iter().enumerate() {
        let assigned_id = ingest_frame(&client, &base_url, &token, idx, vector).await;
        assert_eq!(
            assigned_id, idx,
            "vector ids must stay monotonic and contiguous for identify sequence semantics"
        );
    }

    let perfect_query = dataset[QUERY_START..QUERY_START + QUERY_LEN].to_vec();
    let perfect_payload = identify(&client, &base_url, &token, &perfect_query, 5, 400).await;
    let perfect_results = perfect_payload["data"]["results"]
        .as_array()
        .expect("perfect results array");
    assert!(!perfect_results.is_empty(), "perfect query should match");
    assert_eq!(
        perfect_results[0]["id"].as_u64(),
        Some(QUERY_START as u64),
        "perfect identify should return exact sequence start id"
    );
    let perfect_score = perfect_results[0]["score"].as_f64().expect("perfect score");
    assert!(
        perfect_score >= 0.995,
        "perfect identify score should be near 1.0, got {perfect_score}"
    );

    let mut rng = StdRng::seed_from_u64(0xA11D10BEEFu64);
    let noisy_query = perfect_query
        .iter()
        .map(|vector| add_gaussian_noise(vector, 0.03, &mut rng))
        .collect::<Vec<_>>();
    let noisy_payload = identify(&client, &base_url, &token, &noisy_query, 8, 512).await;
    let noisy_results = noisy_payload["data"]["results"]
        .as_array()
        .expect("noisy results array");
    assert!(
        !noisy_results.is_empty(),
        "noisy query should still produce candidates"
    );
    let noisy_rank = noisy_results
        .iter()
        .position(|row| row["id"].as_u64() == Some(QUERY_START as u64))
        .expect("noisy identify should still include exact start id");
    assert!(
        noisy_rank <= 2,
        "expected exact start id in top-3 for noisy query, got rank {}",
        noisy_rank + 1
    );
    let noisy_score = noisy_results[noisy_rank]["score"]
        .as_f64()
        .expect("noisy score");
    assert!(
        noisy_score >= 0.90,
        "noisy identify score should remain high, got {noisy_score}"
    );

    let mut silent_query = vec![vec![0.0f32; DIM]; QUERY_LEN];
    let mut artifact = deterministic_frame(42_424_242);
    for lane in &mut artifact {
        *lane *= 5.0;
    }
    silent_query[QUERY_LEN / 2] = artifact;
    let silent_payload = identify(&client, &base_url, &token, &silent_query, 5, 512).await;
    let silent_results = silent_payload["data"]["results"]
        .as_array()
        .expect("silent results array");
    assert!(
        silent_results
            .iter()
            .all(|row| row["score"].as_f64().unwrap_or(0.0) < 0.20),
        "silent-anchor query should not return high-confidence false positives"
    );

    stop_server(&mut server).await;
}

#[tokio::test]
async fn identify_matches_multiple_sequences_across_tracks() {
    let _permit = acquire_test_lock().await;
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("identify_protocol_multi_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping identify multi-sequence test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping identify multi-sequence test: failed to spawn server: {}",
                e
            );
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready");

    const TRACK_LEN: usize = 1_536;
    const TRACK_QUERY_LEN: usize = 40;
    const QUERIES_PER_TRACK: usize = 6;
    let track_specs = [
        ("track_alpha.wav", 0xA1A1_A1A1_A1A1_A1A1u64),
        ("track_bravo.wav", 0xB2B2_B2B2_B2B2_B2B2u64),
        ("track_charlie.wav", 0xC3C3_C3C3_C3C3_C3C3u64),
    ];

    struct TrackData {
        source_file: &'static str,
        frames: Vec<Vec<f32>>,
        id_offset: usize,
    }

    let mut tracks = Vec::with_capacity(track_specs.len());
    let mut next_id = 0usize;
    for (source_file, seed) in track_specs {
        let id_offset = next_id;
        let frames = (0..TRACK_LEN)
            .map(|i| deterministic_frame_for_track(seed, i))
            .collect::<Vec<_>>();
        for (local_idx, vector) in frames.iter().enumerate() {
            let assigned = ingest_frame_for_track(
                &client,
                &base_url,
                &token,
                next_id,
                local_idx,
                vector,
                source_file,
            )
            .await;
            assert_eq!(assigned, next_id, "global vector id drifted during ingest");
            next_id += 1;
        }
        tracks.push(TrackData {
            source_file,
            frames,
            id_offset,
        });
    }

    for track in &tracks {
        for probe in 0..QUERIES_PER_TRACK {
            let max_start = TRACK_LEN - TRACK_QUERY_LEN - 1;
            let local_start = 64 + ((probe * 197) % (max_start - 64));
            let query = track.frames[local_start..local_start + TRACK_QUERY_LEN].to_vec();
            let payload = identify(&client, &base_url, &token, &query, 5, 320).await;
            let results = payload["data"]["results"]
                .as_array()
                .expect("multi-sequence results array");
            assert!(
                !results.is_empty(),
                "query for {} should return at least one result",
                track.source_file
            );

            let expected_id = track.id_offset + local_start;
            let top = &results[0];
            assert_eq!(
                top["id"].as_u64(),
                Some(expected_id as u64),
                "top-1 mismatch for {} at local window {}",
                track.source_file,
                local_start
            );
            assert_eq!(
                top["metadata"]["source_file"].as_str(),
                Some(track.source_file),
                "top-1 source_file mismatch for {}",
                track.source_file
            );
            let score = top["score"].as_f64().expect("top score");
            assert!(
                score >= 0.98,
                "expected high-confidence exact match for {}, got {}",
                track.source_file,
                score
            );
        }
    }

    let mut noise_rng = StdRng::seed_from_u64(0xDEC0_DED1_CAFE_BABEu64);
    for track in &tracks {
        let local_start = TRACK_LEN / 2;
        let noisy_query = track.frames[local_start..local_start + TRACK_QUERY_LEN]
            .iter()
            .map(|v| add_gaussian_noise(v, 0.025, &mut noise_rng))
            .collect::<Vec<_>>();
        let payload = identify(&client, &base_url, &token, &noisy_query, 8, 420).await;
        let results = payload["data"]["results"]
            .as_array()
            .expect("noisy multi-sequence results");
        let expected_id = track.id_offset + local_start;
        let rank = results
            .iter()
            .position(|row| row["id"].as_u64() == Some(expected_id as u64))
            .expect("noisy multi-sequence should still include exact id");
        assert!(
            rank <= 2,
            "expected {} noisy match in top-3, got rank {}",
            track.source_file,
            rank + 1
        );
        assert_eq!(
            results[rank]["metadata"]["source_file"].as_str(),
            Some(track.source_file),
            "noisy match source_file mismatch for {}",
            track.source_file
        );
    }

    stop_server(&mut server).await;
}
