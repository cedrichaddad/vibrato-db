use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;

const DIM: usize = 8;

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
        .arg(DIM.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .env("VIBRATO_API_PEPPER", "test-pepper")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("idempotency-limit-e2e")
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

async fn wait_for_ready(base_url: &str, token: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v3/health/ready", base_url);

    for _ in 0..80 {
        if let Ok(resp) = client.get(&ready_url).bearer_auth(token).send().await {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_ingest_rejects_oversized_idempotency_key() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("idempotency_limit_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping idempotency key limit test: localhost bind unavailable");
        return;
    };

    let token = create_api_key(&data_dir).expect("create api key");
    let base_url = format!("http://127.0.0.1:{port}");
    let mut server = start_server(&data_dir, port).await.expect("start serve-v3");
    wait_for_ready(&base_url, &token).await.expect("ready");

    let client = reqwest::Client::new();
    let long_key = "x".repeat(65);
    let payload = serde_json::json!({
        "vector": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
        "metadata": {
            "entity_id": 1,
            "sequence_ts": 123,
            "tags": ["limit-test"],
            "payload_base64": ""
        },
        "idempotency_key": long_key
    });
    let resp = client
        .post(format!("{}/v3/vectors", base_url))
        .bearer_auth(&token)
        .json(&payload)
        .send()
        .await
        .expect("ingest request");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = resp.json().await.expect("json error body");
    let msg = body["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("idempotency key too long"),
        "unexpected message: {msg}"
    );

    stop_server(&mut server).await;
}
