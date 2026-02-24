use std::net::TcpListener;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time::sleep;

const DIM: usize = 4;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn stop_server(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

async fn wait_for_ready_public(child: &mut Child, base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v3/health/ready", base_url);

    for _ in 0..120 {
        if let Ok(Some(status)) = child.try_wait() {
            let mut stderr_text = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut stderr_text).await;
            }
            return Err(format!(
                "server exited before ready: status={} stderr={}",
                status,
                stderr_text.trim()
            ));
        }

        if let Ok(resp) = client.get(&ready_url).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("server did not become ready at {}", ready_url))
}

fn create_api_key(data_dir: &Path, pepper: &str) -> Result<String, String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("fault-injection")
        .arg("--roles")
        .arg("admin,query,ingest")
        .env("VIBRATO_API_PEPPER", pepper)
        .output()
        .map_err(|e| format!("key-create spawn failed: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "key-create failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(token) = line.strip_prefix("token=") {
            return Ok(token.trim().to_string());
        }
    }
    Err(format!("token not found in key-create output: {}", stdout))
}

async fn start_server(
    data_dir: &Path,
    http_port: u16,
    pepper: &str,
    writer_fault_after: Option<usize>,
) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v3")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(http_port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .arg("--public-health-metrics")
        .arg("true")
        .env("VIBRATO_API_PEPPER", pepper)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(after) = writer_fault_after {
        cmd.env(
            "VIBRATO_TEST_FAULT_INGEST_WRITER_PANIC_AFTER_JOBS",
            after.to_string(),
        );
    }
    cmd.spawn()
}

#[cfg(not(debug_assertions))]
#[tokio::test]
async fn serve_v3_exits_without_api_pepper() {
    let Some(port) = reserve_local_port() else {
        eprintln!("skipping: localhost bind unavailable");
        return;
    };
    let dir = tempdir().expect("tempdir");
    let data_dir = dir.path().join("pepper_required");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let mut child = Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("serve-v3")
        .arg("--data-dir")
        .arg(&data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env_remove("VIBRATO_API_PEPPER")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve-v3");

    for _ in 0..60 {
        if let Some(status) = child.try_wait().expect("try_wait") {
            assert!(!status.success(), "serve-v3 unexpectedly succeeded");
            let mut stderr_text = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut stderr_text).await;
            }
            assert!(
                stderr_text.contains("FATAL: VIBRATO_API_PEPPER"),
                "expected fatal pepper message, got: {}",
                stderr_text
            );
            return;
        }
        sleep(Duration::from_millis(50)).await;
    }

    stop_server(&mut child).await;
    panic!("serve-v3 did not exit without VIBRATO_API_PEPPER");
}

#[tokio::test]
async fn ready_returns_503_when_ingest_writer_panics() {
    let Some(port) = reserve_local_port() else {
        eprintln!("skipping: localhost bind unavailable");
        return;
    };

    let dir = tempdir().expect("tempdir");
    let data_dir = dir.path().join("writer_fault");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let pepper = "fault-test-pepper";
    let token = create_api_key(&data_dir, pepper).expect("create api key");

    let mut child = start_server(&data_dir, port, pepper, Some(1))
        .await
        .expect("start serve-v3");
    let base_url = format!("http://127.0.0.1:{}", port);
    if let Err(err) = wait_for_ready_public(&mut child, &base_url).await {
        stop_server(&mut child).await;
        panic!("{err}");
    }

    let client = reqwest::Client::new();
    let ingest_resp = client
        .post(format!("{}/v3/vectors", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vector": [1.0f32, 0.0, 0.0, 0.0],
            "metadata": {
                "entity_id": 1u64,
                "sequence_ts": 1u64,
                "tags": ["fault"],
                "payload_base64": ""
            }
        }))
        .send()
        .await
        .expect("ingest request");
    assert!(
        ingest_resp.status().is_success(),
        "initial ingest should succeed before injected panic, status={}",
        ingest_resp.status()
    );

    let ready_url = format!("{}/v3/health/ready", base_url);
    for _ in 0..120 {
        let resp = client.get(&ready_url).send().await.expect("ready request");
        if resp.status() == StatusCode::SERVICE_UNAVAILABLE {
            let body = resp.text().await.unwrap_or_default();
            assert!(
                body.contains("ingest writer unavailable"),
                "ready body should report writer failure, body={}",
                body
            );
            stop_server(&mut child).await;
            return;
        }
        sleep(Duration::from_millis(100)).await;
    }

    stop_server(&mut child).await;
    panic!("expected /v3/health/ready to become 503 after writer panic");
}
