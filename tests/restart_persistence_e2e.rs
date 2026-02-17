use std::net::TcpListener;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;
use vibrato_db::format_v2::VdbWriterV2;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_path: &Path, port: u16) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve")
        .arg("--data")
        .arg(data_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_health(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let health_url = format!("{}/health", base_url);

    for _ in 0..60 {
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }

    Err(format!("server did not become healthy at {}", health_url))
}

async fn stop_server(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

#[tokio::test]
async fn test_persistence_integrity_across_process_restart() {
    let dir = tempdir().expect("tempdir");
    let data_path = dir.path().join("index.vdb");

    let writer = VdbWriterV2::new_raw(&data_path, 2).expect("create empty v2");
    writer.finish().expect("finish empty v2");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping restart persistence test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_path, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping restart persistence test: failed to spawn server process: {}",
                e
            );
            return;
        }
    };
    wait_for_health(&base_url).await.expect("initial health");

    let ingest_body = serde_json::json!({
        "vector": [0.1, 0.2],
        "metadata": {
            "source_file": "demo/snare.wav",
            "bpm": 128.0,
            "tags": ["drums", "snare"]
        }
    });

    let ingest_resp = client
        .post(format!("{}/ingest", base_url))
        .json(&ingest_body)
        .send()
        .await
        .expect("ingest request");
    assert_eq!(ingest_resp.status(), StatusCode::CREATED);

    let flush_resp = client
        .post(format!("{}/flush", base_url))
        .send()
        .await
        .expect("flush request");
    if flush_resp.status() == StatusCode::ACCEPTED {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if std::time::Instant::now() > deadline {
                panic!("flush timed out");
            }
            let status_resp = client
                .get(format!("{}/flush/status", base_url))
                .send()
                .await
                .expect("flush status request");
            assert_eq!(status_resp.status(), StatusCode::OK);
            let status: serde_json::Value = status_resp.json().await.expect("flush status json");
            match status["state"].as_str().unwrap_or("unknown") {
                "completed" | "idle" => break,
                "failed" => panic!("flush failed: {:?}", status),
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
    } else {
        assert_eq!(flush_resp.status(), StatusCode::OK);
    }

    stop_server(&mut server).await;

    let mut restarted = start_server(&data_path, port)
        .await
        .expect("restart server process");
    wait_for_health(&base_url).await.expect("restart health");

    let search_body = serde_json::json!({
        "vector": [0.1, 0.2],
        "k": 1,
        "ef": 20
    });

    let search_resp = client
        .post(format!("{}/search", base_url))
        .json(&search_body)
        .send()
        .await
        .expect("search request");
    assert_eq!(search_resp.status(), StatusCode::OK);

    let payload: serde_json::Value = search_resp.json().await.expect("search json");
    let first = payload["results"]
        .as_array()
        .and_then(|arr| arr.first())
        .expect("first search result");
    assert_eq!(first["id"].as_u64().expect("result id"), 0);
    assert_eq!(first["metadata"]["source_file"], "demo/snare.wav");
    assert_eq!(first["metadata"]["bpm"].as_f64().expect("bpm"), 128.0);
    assert_eq!(first["metadata"]["tags"][0], "drums");
    assert_eq!(first["metadata"]["tags"][1], "snare");

    stop_server(&mut restarted).await;
}
