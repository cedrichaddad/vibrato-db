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
        .arg("1")
        .arg("--compaction-interval-secs")
        .arg("3")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);

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
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("golden-ci")
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
async fn test_golden_set_day_in_life_restart_integrity() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("prod_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let token = create_api_key(&data_dir).expect("create api key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping golden v2 test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!("skipping golden v2 test: failed to spawn server: {}", e);
            return;
        }
    };

    wait_for_ready(&base_url).await.expect("server ready");

    for i in 0..100usize {
        let v0 = i as f32 / 100.0;
        let v1 = 1.0 - v0;
        let tag = if i % 2 == 0 { "snare" } else { "kick" };

        let body = serde_json::json!({
            "vector": [v0, v1],
            "metadata": {
                "source_file": format!("sample_{}.wav", i),
                "start_time_ms": i * 10,
                "duration_ms": 500,
                "bpm": 120.0 + (i % 10) as f32,
                "tags": ["drums", tag]
            },
            "idempotency_key": format!("golden-{}", i)
        });

        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await
            .expect("ingest request");
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    let query_body = serde_json::json!({
        "vector": [0.42, 0.58],
        "k": 5,
        "ef": 50,
        "include_metadata": true,
        "filter": {
            "tags_all": ["drums"],
            "tags_any": ["snare"]
        }
    });

    let query_resp = client
        .post(format!("{}/v2/query", base_url))
        .bearer_auth(&token)
        .json(&query_body)
        .send()
        .await
        .expect("query request");
    assert_eq!(query_resp.status(), StatusCode::OK);

    let checkpoint_resp = client
        .post(format!("{}/v2/admin/checkpoint", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("checkpoint request");
    assert_eq!(checkpoint_resp.status(), StatusCode::OK);

    let snapshot_resp = client
        .post(format!("{}/v2/admin/snapshot", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("snapshot request");
    assert_eq!(snapshot_resp.status(), StatusCode::OK);

    stop_server(&mut server).await;

    let mut restarted = start_server(&data_dir, port)
        .await
        .expect("restart server process");
    wait_for_ready(&base_url)
        .await
        .expect("server ready after restart");

    let stats_resp = client
        .get(format!("{}/v2/admin/stats", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("stats request");
    assert_eq!(stats_resp.status(), StatusCode::OK);

    let stats: serde_json::Value = stats_resp.json().await.expect("stats json");
    let total_vectors = stats["data"]["total_vectors"]
        .as_u64()
        .expect("total_vectors in stats");
    assert_eq!(total_vectors, 100);

    let query_resp_after = client
        .post(format!("{}/v2/query", base_url))
        .bearer_auth(&token)
        .json(&query_body)
        .send()
        .await
        .expect("query after restart");
    assert_eq!(query_resp_after.status(), StatusCode::OK);

    let payload: serde_json::Value = query_resp_after.json().await.expect("query json");
    let first = payload["data"]["results"]
        .as_array()
        .and_then(|arr| arr.first())
        .expect("first query result");

    let tags = first["metadata"]["tags"].as_array().expect("tags array");
    assert!(tags.iter().any(|t| t == "drums"));

    stop_server(&mut restarted).await;
}
