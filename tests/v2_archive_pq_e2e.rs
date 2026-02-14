use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;
use vibrato_core::format_v2::VdbHeaderV2;

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
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    for _ in 0..120 {
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
        .arg("archive-pq")
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

async fn ingest_batch(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    prefix: &str,
    start: usize,
    count: usize,
) {
    for i in 0..count {
        let id = start + i;
        let v = id as f32 / 300.0;
        let body = serde_json::json!({
            "vector": [v, 1.0 - v],
            "metadata": {
                "source_file": format!("{}-{}.wav", prefix, id),
                "start_time_ms": id * 5,
                "duration_ms": 220,
                "bpm": 124.0,
                "tags": ["archive", "pq", prefix]
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

async fn run_admin_job(client: &reqwest::Client, base_url: &str, token: &str, path: &str) {
    let resp = client
        .post(format!("{}/{}", base_url, path))
        .bearer_auth(token)
        .send()
        .await
        .expect("admin request");
    assert_eq!(resp.status(), StatusCode::OK, "job path {}", path);
}

#[tokio::test]
async fn level2_archive_segments_are_pq_encoded_and_queryable() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("archive_pq_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping archive pq test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!("skipping archive pq test: failed to spawn server: {}", e);
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready");

    ingest_batch(&client, &base_url, &token, "a", 0, 64).await;
    run_admin_job(&client, &base_url, &token, "v2/admin/checkpoint").await;
    ingest_batch(&client, &base_url, &token, "b", 64, 64).await;
    run_admin_job(&client, &base_url, &token, "v2/admin/checkpoint").await;
    run_admin_job(&client, &base_url, &token, "v2/admin/compact").await; // L1 #1

    ingest_batch(&client, &base_url, &token, "c", 128, 64).await;
    run_admin_job(&client, &base_url, &token, "v2/admin/checkpoint").await;
    ingest_batch(&client, &base_url, &token, "d", 192, 64).await;
    run_admin_job(&client, &base_url, &token, "v2/admin/checkpoint").await;
    run_admin_job(&client, &base_url, &token, "v2/admin/compact").await; // L1 #2

    run_admin_job(&client, &base_url, &token, "v2/admin/compact").await; // L2 from two L1s

    let query = serde_json::json!({
        "vector": [0.22, 0.78],
        "k": 10,
        "ef": 50,
        "search_tier": "archive",
        "include_metadata": true,
        "filter": {"tags_any": ["pq"]}
    });
    let query_resp = client
        .post(format!("{}/v2/query", base_url))
        .bearer_auth(&token)
        .json(&query)
        .send()
        .await
        .expect("archive query");
    assert_eq!(query_resp.status(), StatusCode::OK);
    let payload: serde_json::Value = query_resp.json().await.expect("query payload");
    let results = payload["data"]["results"]
        .as_array()
        .expect("results array");
    assert!(
        !results.is_empty(),
        "archive query should return at least one candidate"
    );

    stop_server(&mut server).await;

    let catalog_path = data_dir.join("catalog.sqlite3");
    let sql_out = std::process::Command::new("sqlite3")
        .arg("-json")
        .arg(&catalog_path)
        .arg("SELECT path FROM segments WHERE state='active' AND level=2 ORDER BY row_count DESC LIMIT 1;")
        .output()
        .expect("sqlite3 query");
    assert!(
        sql_out.status.success(),
        "sqlite query failed: {}",
        String::from_utf8_lossy(&sql_out.stderr)
    );
    let rows: serde_json::Value = serde_json::from_slice(&sql_out.stdout).expect("parse json rows");
    let path = rows
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|r| r.get("path"))
        .and_then(|v| v.as_str())
        .expect("active level-2 segment path");

    let bytes = std::fs::read(path).expect("read level2 segment");
    let header = VdbHeaderV2::from_bytes(&bytes).expect("parse v2 header");
    assert!(
        header.is_pq_enabled(),
        "archive level-2 segment must be pq-enabled"
    );
    assert!(header.pq_subspaces > 0, "pq_subspaces should be set");
    assert!(
        header.codebook_offset > 0,
        "pq codebook offset should be set"
    );
}
