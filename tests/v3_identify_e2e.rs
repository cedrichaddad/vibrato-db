use std::f32::consts::PI;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;
use vibrato_db::prod::model::encode_payload_base64;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, port: u16) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v3")
        .env("VIBRATO_API_PEPPER", "test-pepper")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--dim")
        .arg("2")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .arg("--admin-checkpoint-cooldown-secs")
        .arg("0")
        .arg("--admin-compaction-cooldown-secs")
        .arg("0")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v3/health/ready", base_url);
    for _ in 0..80 {
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
        .env("VIBRATO_API_PEPPER", "test-pepper")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("identify-e2e")
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

fn make_frame(base_phase: f32, idx: usize) -> [f32; 2] {
    let phase = base_phase + idx as f32 * (PI / 20.0);
    [phase.cos(), phase.sin()]
}

async fn ingest_track(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    source_file: &str,
    base_phase: f32,
    start_idx: usize,
    count: usize,
) {
    for i in 0..count {
        let id = start_idx + i;
        let frame = make_frame(base_phase, i);
        let payload = format!("{source_file}-frame-{i}");
        let body = serde_json::json!({
            "vector": frame,
            "metadata": {
                "entity_id": id as u64,
                "sequence_ts": (i * 100) as u64,
                "payload_base64": encode_payload_base64(payload.as_bytes()),
                "tags": ["identify", source_file]
            },
            "idempotency_key": format!("{}-{}", source_file, id)
        });
        let resp = client
            .post(format!("{}/v3/vectors", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("ingest");
        assert_eq!(resp.status(), StatusCode::CREATED);
    }
}

async fn checkpoint(client: &reqwest::Client, base_url: &str, token: &str) {
    let resp = client
        .post(format!("{}/v3/admin/checkpoint", base_url))
        .bearer_auth(token)
        .send()
        .await
        .expect("checkpoint");
    assert_eq!(resp.status(), StatusCode::OK);
    let payload: serde_json::Value = resp.json().await.expect("checkpoint payload");
    assert_eq!(
        payload["data"]["state"].as_str(),
        Some("completed"),
        "checkpoint did not complete: {}",
        payload
    );
}

#[tokio::test]
async fn identify_matches_sequence_and_handles_segment_boundaries() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("identify_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping identify test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!("skipping identify test: failed to spawn server: {}", e);
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready");

    ingest_track(&client, &base_url, &token, "track_a.wav", 0.0, 0, 8).await;
    checkpoint(&client, &base_url, &token).await;

    ingest_track(&client, &base_url, &token, "track_b.wav", PI * 0.6, 8, 8).await;
    checkpoint(&client, &base_url, &token).await;

    let identify_query: Vec<Vec<f32>> = (1..7)
        .map(|i| {
            let frame = make_frame(0.0, i);
            vec![frame[0], frame[1]]
        })
        .collect();
    let identify_resp = client
        .post(format!("{}/v3/identify", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vectors": identify_query,
            "k": 3,
            "ef": 200,
            "include_metadata": true,
            "search_tier": "active"
        }))
        .send()
        .await
        .expect("identify");
    let identify_status = identify_resp.status();
    let identify_body = identify_resp.text().await.expect("identify body");
    assert_eq!(
        identify_status,
        StatusCode::OK,
        "identify status={} body={}",
        identify_status,
        identify_body
    );
    let identify_payload: serde_json::Value =
        serde_json::from_str(&identify_body).expect("identify json");
    let results = identify_payload["data"]["results"]
        .as_array()
        .expect("results array");
    assert!(
        !results.is_empty(),
        "identify should return at least one match"
    );
    assert_eq!(results[0]["id"].as_u64(), Some(1));
    assert_eq!(
        results[0]["metadata"]["entity_id"].as_u64(),
        Some(1),
        "expected entity_id to match vector id"
    );
    assert_eq!(
        results[0]["metadata"]["sequence_ts"].as_u64(),
        Some(100),
        "expected sequence_ts for frame idx=1"
    );
    let tags = results[0]["metadata"]["tags"]
        .as_array()
        .expect("metadata.tags array");
    assert!(
        tags.iter().any(|v| v.as_str() == Some("track_a.wav")),
        "expected track tag in metadata.tags: {:?}",
        tags
    );
    let expected_payload_b64 = encode_payload_base64("track_a.wav-frame-1".as_bytes());
    assert_eq!(
        results[0]["metadata"]["payload_base64"].as_str(),
        Some(expected_payload_b64.as_str())
    );

    let boundary_query: Vec<Vec<f32>> = (5..8)
        .map(|i| {
            let frame = make_frame(0.0, i);
            vec![frame[0], frame[1]]
        })
        .chain((0..3).map(|i| {
            let frame = make_frame(PI * 0.6, i);
            vec![frame[0], frame[1]]
        }))
        .collect();
    let boundary_resp = client
        .post(format!("{}/v3/identify", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vectors": boundary_query,
            "k": 8,
            "ef": 200,
            "include_metadata": true,
            "search_tier": "active"
        }))
        .send()
        .await
        .expect("identify boundary");
    let boundary_status = boundary_resp.status();
    let boundary_body = boundary_resp.text().await.expect("boundary body");
    assert_eq!(
        boundary_status,
        StatusCode::OK,
        "identify boundary status={} body={}",
        boundary_status,
        boundary_body
    );
    let boundary_payload: serde_json::Value =
        serde_json::from_str(&boundary_body).expect("boundary json");
    let boundary_results = boundary_payload["data"]["results"]
        .as_array()
        .expect("boundary results array");
    for result in boundary_results {
        let id = result["id"].as_u64().expect("result id") as usize;
        let valid = (0..=2).contains(&id) || (8..=10).contains(&id);
        assert!(
            valid,
            "identify returned cross-segment start_id {}, expected in [0..=2] or [8..=10]",
            id
        );
    }

    stop_server(&mut server).await;
}
