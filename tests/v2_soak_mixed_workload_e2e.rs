use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::process::Stdio;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;

const DIM: usize = 16;

fn process_rss_kb(pid: u32) -> Option<u64> {
    let output = StdCommand::new("ps")
        .arg("-o")
        .arg("rss=")
        .arg("-p")
        .arg(pid.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
}

fn process_fd_count(pid: u32) -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let fd_dir = format!("/proc/{}/fd", pid);
        if let Ok(entries) = std::fs::read_dir(fd_dir) {
            return Some(entries.count());
        }
    }

    let output = StdCommand::new("lsof")
        .arg("-p")
        .arg(pid.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let lines = String::from_utf8_lossy(&output.stdout);
    Some(lines.lines().skip(1).count())
}

fn count_tmp_vdb_files(tmp_dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(tmp_dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .file_name()
                .and_then(|v| v.to_str())
                .map(|name| name.ends_with(".vdb.tmp"))
                .unwrap_or(false)
        })
        .count()
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
        .arg("15")
        .arg("--compaction-interval-secs")
        .arg("60")
        .arg("--public-health-metrics")
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    for _ in 0..200 {
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
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("soak")
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

fn normalized_vector(seed: u64, idx: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)));
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        v.push(rng.gen::<f32>() * 2.0 - 1.0);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "72h soak harness; run on dedicated runner with VIBRATO_SOAK_SECS=259200"]
async fn mixed_workload_soak() {
    let soak_secs = std::env::var("VIBRATO_SOAK_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(259_200);
    let soak_seed = std::env::var("VIBRATO_SOAK_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let sample_interval_secs = std::env::var("VIBRATO_SOAK_SAMPLE_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(60)
        .max(5);
    let target_ops_per_sec = std::env::var("VIBRATO_SOAK_TARGET_OPS_PER_SEC")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500)
        .max(1);
    let max_rss_growth_pct = std::env::var("VIBRATO_SOAK_MAX_RSS_GROWTH_PCT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(35.0);
    let max_fd_growth = std::env::var("VIBRATO_SOAK_MAX_FD_GROWTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);
    let max_tmp_vdb_files = std::env::var("VIBRATO_SOAK_MAX_TMP_VDB_FILES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("soak_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping soak test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
        .expect("build client");
    let mut rng = StdRng::seed_from_u64(soak_seed);

    let mut server = start_server(&data_dir, port)
        .await
        .expect("start soak server");
    wait_for_ready(&base_url).await.expect("ready");
    let server_pid = server.id().unwrap_or(0);

    let start = Instant::now();
    let mut next_sample_at = Instant::now() + Duration::from_secs(sample_interval_secs);
    let mut op_idx = 0usize;
    let mut ingested = 0usize;
    let mut known_vectors: Vec<Vec<f32>> = Vec::new();
    let mut rss_samples_kb = Vec::<u64>::new();
    let mut fd_samples = Vec::<usize>::new();

    while start.elapsed().as_secs() < soak_secs {
        let roll = rng.gen::<u8>() % 100;
        if roll < 60 {
            let vec = normalized_vector(soak_seed, op_idx);
            let body = serde_json::json!({
                "vector": vec,
                "metadata": {
                    "source_file": format!("soak_{op_idx}.wav"),
                    "start_time_ms": op_idx * 10,
                    "duration_ms": 80,
                    "bpm": 120.0 + ((op_idx % 20) as f32),
                    "tags": ["soak", "mixed"]
                },
                "idempotency_key": format!("soak-{op_idx}")
            });
            let resp = client
                .post(format!("{}/v2/vectors", base_url))
                .bearer_auth(&token)
                .json(&body)
                .send()
                .await
                .expect("ingest");
            assert_eq!(resp.status(), StatusCode::CREATED);
            known_vectors.push(normalized_vector(soak_seed, op_idx));
            ingested += 1;
        } else if roll < 90 {
            if let Some(vec) = known_vectors.get(rng.gen_range(0..known_vectors.len().max(1))) {
                let body = serde_json::json!({
                    "vector": vec,
                    "k": 10,
                    "ef": 50,
                    "include_metadata": true
                });
                let resp = client
                    .post(format!("{}/v2/query", base_url))
                    .bearer_auth(&token)
                    .json(&body)
                    .send()
                    .await
                    .expect("query");
                assert_eq!(resp.status(), StatusCode::OK);
            }
        } else if roll < 97 {
            let resp = client
                .post(format!("{}/v2/admin/checkpoint", base_url))
                .bearer_auth(&token)
                .send()
                .await
                .expect("checkpoint");
            assert_eq!(resp.status(), StatusCode::OK);
        } else {
            let resp = client
                .post(format!("{}/v2/admin/compact", base_url))
                .bearer_auth(&token)
                .send()
                .await
                .expect("compact");
            assert_eq!(resp.status(), StatusCode::OK);
        }

        if op_idx % 200 == 0 {
            let ready = client
                .get(format!("{}/v2/health/ready", base_url))
                .send()
                .await
                .expect("ready check");
            assert_eq!(
                ready.status(),
                StatusCode::OK,
                "server became unready during soak"
            );
        }
        if server_pid > 0 && Instant::now() >= next_sample_at {
            if let Some(rss) = process_rss_kb(server_pid) {
                rss_samples_kb.push(rss);
            }
            if let Some(fd_count) = process_fd_count(server_pid) {
                fd_samples.push(fd_count);
            }
            next_sample_at += Duration::from_secs(sample_interval_secs);
        }

        op_idx += 1;
        let expected_elapsed = Duration::from_secs_f64(op_idx as f64 / target_ops_per_sec as f64);
        if expected_elapsed > start.elapsed() {
            sleep(expected_elapsed - start.elapsed()).await;
        }
    }

    let stats = client
        .get(format!("{}/v2/admin/stats", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("stats");
    assert_eq!(stats.status(), StatusCode::OK);
    let payload: serde_json::Value = stats.json().await.expect("stats json");
    let total_vectors = payload["data"]["total_vectors"].as_u64().unwrap_or(0);
    assert_eq!(
        total_vectors,
        ingested as u64,
        "stats total_vectors should equal ingested writes in append-only soak (stats={}, ingested={})",
        total_vectors,
        ingested
    );

    if rss_samples_kb.len() >= 4 {
        let first = rss_samples_kb[0] as f64;
        let max_seen = *rss_samples_kb.iter().max().unwrap_or(&rss_samples_kb[0]) as f64;
        let growth_pct = if first <= 0.0 {
            0.0
        } else {
            ((max_seen - first) / first) * 100.0
        };
        assert!(
            growth_pct <= max_rss_growth_pct,
            "rss growth exceeded threshold: first_kb={} max_kb={} growth_pct={:.2} limit_pct={:.2}",
            first,
            max_seen,
            growth_pct,
            max_rss_growth_pct
        );
    }

    if fd_samples.len() >= 4 {
        let first = fd_samples[0];
        let max_seen = *fd_samples.iter().max().unwrap_or(&first);
        assert!(
            max_seen <= first.saturating_add(max_fd_growth),
            "fd growth exceeded threshold: first={} max={} limit={}",
            first,
            max_seen,
            first.saturating_add(max_fd_growth)
        );
    }

    let tmp_vdb_files = count_tmp_vdb_files(&data_dir.join("tmp"));
    assert!(
        tmp_vdb_files <= max_tmp_vdb_files,
        "tmp segment leak suspected: tmp_vdb_files={} limit={}",
        tmp_vdb_files,
        max_tmp_vdb_files
    );

    stop_server(&mut server).await;
}
