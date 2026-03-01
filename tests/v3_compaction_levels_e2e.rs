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
        .arg("compact-levels")
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

async fn ingest_range(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    prefix: &str,
    start: usize,
    count: usize,
) {
    for i in 0..count {
        let id = start + i;
        let payload = format!("{prefix}-payload-{id}");
        let body = serde_json::json!({
            "vector": [id as f32 / 100.0, 1.0 - (id as f32 / 100.0)],
            "metadata": {
                "entity_id": id as u64,
                "sequence_ts": (id * 10) as u64,
                "payload_base64": encode_payload_base64(payload.as_bytes()),
                "tags": ["archive", prefix]
            },
            "idempotency_key": format!("{}-{}", prefix, id)
        });
        let resp = client
            .post(format!("{}/v3/vectors", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("ingest request");
        assert_eq!(resp.status(), StatusCode::CREATED);
    }
}

async fn run_admin_job(client: &reqwest::Client, base_url: &str, token: &str, path: &str) {
    let resp = client
        .post(format!("{}/{}", base_url, path))
        .bearer_auth(token)
        .send()
        .await
        .expect("admin request");
    assert_eq!(resp.status(), StatusCode::OK, "job path {}", path);
    let payload: serde_json::Value = resp.json().await.expect("admin response payload");
    let state = payload["data"]["state"].as_str().unwrap_or("<missing>");
    assert_eq!(
        state, "completed",
        "job path {} returned state={} details={}",
        path, state, payload["data"]["details"]
    );
}

#[tokio::test]
async fn test_compaction_builds_level2_archive_segment() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("compact_levels_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping compaction levels test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping compaction levels test: failed to spawn server: {}",
                e
            );
            return;
        }
    };

    wait_for_ready(&base_url).await.expect("server ready");

    ingest_range(&client, &base_url, &token, "batch-a", 0, 20).await;
    run_admin_job(&client, &base_url, &token, "v3/admin/checkpoint").await;

    ingest_range(&client, &base_url, &token, "batch-b", 20, 20).await;
    run_admin_job(&client, &base_url, &token, "v3/admin/checkpoint").await;
    run_admin_job(&client, &base_url, &token, "v3/admin/compact").await; // -> level 1

    ingest_range(&client, &base_url, &token, "batch-c", 40, 20).await;
    run_admin_job(&client, &base_url, &token, "v3/admin/checkpoint").await;

    ingest_range(&client, &base_url, &token, "batch-d", 60, 20).await;
    run_admin_job(&client, &base_url, &token, "v3/admin/checkpoint").await;
    run_admin_job(&client, &base_url, &token, "v3/admin/compact").await; // -> level 1
    run_admin_job(&client, &base_url, &token, "v3/admin/compact").await; // level1 + level1 -> level 2

    let archive_query = serde_json::json!({
        "vector": [0.03, 0.97],
        "k": 5,
        "ef": 50,
        "search_tier": "archive",
        "include_metadata": true,
        "filter": {"tags_any": ["archive"]}
    });
    let archive_resp = client
        .post(format!("{}/v3/query", base_url))
        .bearer_auth(&token)
        .json(&archive_query)
        .send()
        .await
        .expect("archive query");
    assert_eq!(archive_resp.status(), StatusCode::OK);

    stop_server(&mut server).await;

    let catalog_path = data_dir.join("catalog.sqlite3");
    let sql_out = std::process::Command::new("sqlite3")
        .arg("-json")
        .arg(&catalog_path)
        .arg("SELECT COUNT(*) AS n FROM segments WHERE state='active' AND level=2;")
        .output()
        .expect("run sqlite3");
    assert!(
        sql_out.status.success(),
        "sqlite3 failed: {}",
        String::from_utf8_lossy(&sql_out.stderr)
    );
    let rows: serde_json::Value =
        serde_json::from_slice(&sql_out.stdout).expect("parse sqlite json");
    let n = rows
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|row| row.get("n"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    assert!(
        n >= 1,
        "expected at least one active level-2 segment, got {}",
        n
    );
}
