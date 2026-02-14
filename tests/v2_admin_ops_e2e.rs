use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(
    data_dir: &Path,
    port: u16,
    public_health_metrics: bool,
) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v2")
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
        .arg("60")
        .arg("--compaction-interval-secs")
        .arg("600")
        .arg("--public-health-metrics")
        .arg(if public_health_metrics {
            "true"
        } else {
            "false"
        })
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str, token: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);

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
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("admin-ops")
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

#[tokio::test]
async fn test_ops_health_auth_and_replay_to_lsn() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("ops_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let token = create_api_key(&data_dir).expect("create api key");
    let Some(port) = reserve_local_port() else {
        eprintln!("skipping ops v2 test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port, false).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!("skipping ops v2 test: failed to spawn server: {}", e);
            return;
        }
    };

    let mut unauth_status = None;
    for _ in 0..60 {
        if let Ok(resp) = client
            .get(format!("{}/v2/health/live", base_url))
            .send()
            .await
        {
            unauth_status = Some(resp.status());
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(unauth_status, Some(StatusCode::UNAUTHORIZED));

    wait_for_ready(&base_url, &token)
        .await
        .expect("ready with token");

    for i in 0..20usize {
        let body = serde_json::json!({
            "vector": [i as f32 / 20.0, 1.0 - (i as f32 / 20.0)],
            "metadata": {
                "source_file": format!("take_{}.wav", i),
                "start_time_ms": i * 10,
                "duration_ms": 300,
                "bpm": 120.0,
                "tags": ["drums"]
            },
            "idempotency_key": format!("ops-{}", i)
        });
        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await
            .expect("ingest");
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    stop_server(&mut server).await;

    let replay_out = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("replay-to-lsn")
        .arg("--data-dir")
        .arg(&data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--target-lsn")
        .arg("10")
        .output()
        .expect("run replay-to-lsn");
    assert!(
        replay_out.status.success(),
        "replay-to-lsn failed: {}",
        String::from_utf8_lossy(&replay_out.stderr)
    );

    let mut restarted = start_server(&data_dir, port, false)
        .await
        .expect("restart server");
    wait_for_ready(&base_url, &token)
        .await
        .expect("ready after replay");

    let stats_resp = client
        .get(format!("{}/v2/admin/stats", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("stats");
    assert_eq!(stats_resp.status(), StatusCode::OK);
    let stats: serde_json::Value = stats_resp.json().await.expect("stats json");
    let total_vectors = stats["data"]["total_vectors"]
        .as_u64()
        .expect("total_vectors");
    assert_eq!(total_vectors, 11);

    stop_server(&mut restarted).await;
}
