use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time::sleep;

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
        .arg("2")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn()
}

async fn wait_for_ready(
    child: &mut Child,
    base_url: &str,
    token: Option<&str>,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    let mut last_status = String::new();
    let mut last_body = String::new();
    for _ in 0..180 {
        if let Ok(Some(status)) = child.try_wait() {
            let mut stderr_text = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut stderr_text).await;
            }
            return Err(format!(
                "server exited before ready: status={} url={} stderr={}",
                status,
                ready_url,
                stderr_text.trim()
            ));
        }

        let mut req = client.get(&ready_url);
        if let Some(token) = token {
            req = req.bearer_auth(token);
        }
        if let Ok(resp) = req.send().await {
            let status = resp.status();
            if status == StatusCode::OK {
                return Ok(());
            }
            last_status = status.to_string();
            last_body = resp.text().await.unwrap_or_default();
            if last_body.len() > 300 {
                last_body.truncate(300);
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!(
        "server did not become ready at {} (last_status={} last_body={})",
        ready_url, last_status, last_body
    ))
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
        .arg("snapshot-restore")
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

async fn ingest_count(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    prefix: &str,
    start: usize,
    count: usize,
) {
    for i in 0..count {
        let id = start + i;
        let body = serde_json::json!({
            "vector": [id as f32 / 100.0, 1.0 - (id as f32 / 100.0)],
            "metadata": {
                "source_file": format!("{}-{}.wav", prefix, id),
                "start_time_ms": id * 5,
                "duration_ms": 200,
                "bpm": 120.0,
                "tags": ["snapshot", prefix]
            },
            "idempotency_key": format!("{}-{}", prefix, id)
        });
        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("ingest request");
        assert_eq!(resp.status(), StatusCode::CREATED);
    }
}

fn run_snapshot_create(data_dir: &Path) -> anyhow::Result<PathBuf> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("snapshot-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--collection")
        .arg("default")
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "snapshot-create failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(path) = line.strip_prefix("snapshot_dir=") {
            return Ok(PathBuf::from(path.trim()));
        }
    }
    anyhow::bail!("snapshot_dir not found in output: {}", stdout)
}

fn run_snapshot_restore(data_dir: &Path, snapshot_dir: &Path) -> anyhow::Result<()> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("snapshot-restore")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--snapshot-dir")
        .arg(snapshot_dir)
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "snapshot-restore failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

#[tokio::test]
async fn snapshot_restore_reverts_catalog_and_segments_to_snapshot_point() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("snapshot_restore_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let token = create_api_key(&data_dir).expect("create key");
    let Some(port) = reserve_local_port() else {
        eprintln!("skipping snapshot restore test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping snapshot restore test: failed to spawn server: {}",
                e
            );
            return;
        }
    };
    wait_for_ready(&mut server, &base_url, Some(&token))
        .await
        .expect("ready");

    ingest_count(&client, &base_url, &token, "pre", 0, 30).await;
    let checkpoint = client
        .post(format!("{}/v2/admin/checkpoint", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("checkpoint");
    assert_eq!(checkpoint.status(), StatusCode::OK);
    stop_server(&mut server).await;

    let snapshot_dir = run_snapshot_create(&data_dir).expect("create snapshot");

    let mut server2 = start_server(&data_dir, port).await.expect("restart server");
    wait_for_ready(&mut server2, &base_url, Some(&token))
        .await
        .expect("ready after restart");
    ingest_count(&client, &base_url, &token, "post", 30, 10).await;
    let checkpoint2 = client
        .post(format!("{}/v2/admin/checkpoint", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("checkpoint second");
    assert_eq!(checkpoint2.status(), StatusCode::OK);
    stop_server(&mut server2).await;

    run_snapshot_restore(&data_dir, &snapshot_dir).expect("restore snapshot");

    let mut server3 = start_server(&data_dir, port)
        .await
        .expect("restart after restore");
    wait_for_ready(&mut server3, &base_url, Some(&token))
        .await
        .expect("ready after snapshot restore");
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
    assert_eq!(
        total_vectors, 30,
        "snapshot restore should roll back post-snapshot ingests"
    );
    stop_server(&mut server3).await;
}
