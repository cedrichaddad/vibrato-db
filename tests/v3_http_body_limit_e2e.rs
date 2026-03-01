use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use reqwest::header::{CONTENT_TYPE, EXPECT};
use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::time::sleep;

fn can_bind_localhost() -> bool {
    std::net::TcpListener::bind("127.0.0.1:0").is_ok()
}

fn pick_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
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
        .arg("16")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v3/health/live", base_url);
    for _ in 0..80 {
        if let Ok(resp) = client.get(&ready_url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("server did not become live at {}", ready_url))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_body_limit_rejects_oversized_payload() {
    if !can_bind_localhost() {
        eprintln!("skipping http body limit test: localhost bind unavailable");
        return;
    }

    let dir = tempdir().expect("tempdir");
    let port = pick_port();
    let base_url = format!("http://127.0.0.1:{}", port);

    let mut server = start_server(dir.path(), port)
        .await
        .expect("start serve-v3");
    wait_for_ready(&base_url).await.expect("ready");

    let client = reqwest::Client::new();
    // Send a syntactically valid JSON payload with explicit content type so the
    // request is rejected by HTTP body-size limits (413), not by early media-type
    // or decode-path rejection that can race into a transport reset.
    let oversized_pad = "a".repeat(65 * 1024 * 1024);
    let oversized_json = format!(
        r#"{{"vector":[0.0],"k":1,"ef":1,"include_metadata":false,"pad":"{}"}}"#,
        oversized_pad
    );
    let resp = client
        .post(format!("{}/v3/query", base_url))
        .header(CONTENT_TYPE, "application/json")
        .header(EXPECT, "100-continue")
        .body(oversized_json)
        .send()
        .await
        .expect("oversized request");
    assert_eq!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "expected payload-too-large for >64MiB body limit"
    );

    let _ = server.kill();
    let _ = server.wait();
}
