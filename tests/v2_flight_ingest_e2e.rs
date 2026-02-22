use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, StringArray};
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, PutResult};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time::sleep;

const DIM: usize = 8;

#[derive(Clone)]
struct FlightRow {
    vector: Vec<f32>,
    metadata_json: Option<String>,
    idempotency_key: Option<String>,
}

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, http_port: u16, flight_port: u16) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v2")
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
        .arg("--flight-host")
        .arg("127.0.0.1")
        .arg("--flight-port")
        .arg(flight_port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str, token: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);

    for _ in 0..100 {
        if let Ok(resp) = client.get(&ready_url).bearer_auth(token).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }

    Err(format!("server did not become ready at {}", ready_url))
}

async fn wait_for_flight_ready(endpoint: &str) -> Result<(), String> {
    for _ in 0..100 {
        let channel = tonic::transport::Endpoint::from_shared(endpoint.to_string())
            .map_err(|e| format!("invalid flight endpoint: {e}"))?;
        if channel.connect().await.is_ok() {
            return Ok(());
        }
        sleep(Duration::from_millis(50)).await;
    }
    Err(format!(
        "flight endpoint did not become ready at {}",
        endpoint
    ))
}

async fn stop_server(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

async fn stop_server_with_logs(child: &mut Child) {
    let mut stderr_handle = child.stderr.take();
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    if let Some(mut stderr) = stderr_handle.take() {
        let mut log = String::new();
        let _ = tokio::time::timeout(Duration::from_secs(2), stderr.read_to_string(&mut log)).await;
        if !log.trim().is_empty() {
            eprintln!("\n=== SERVER STDERR (LAST 20 LINES) ===");
            let lines = log.lines().collect::<Vec<_>>();
            let start = lines.len().saturating_sub(20);
            for line in &lines[start..] {
                eprintln!("{}", line);
            }
            eprintln!("====================================\n");
        }
    }
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("flight-e2e")
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

fn build_batch(rows: &[FlightRow], dim: usize) -> RecordBatch {
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.iter()
            .map(|row| Some(row.vector.iter().copied().map(Some).collect::<Vec<_>>())),
        dim as i32,
    );
    let metadata_json = StringArray::from(
        rows.iter()
            .map(|row| row.metadata_json.as_deref())
            .collect::<Vec<_>>(),
    );
    let idempotency_key = StringArray::from(
        rows.iter()
            .map(|row| row.idempotency_key.as_deref())
            .collect::<Vec<_>>(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new("metadata_json", DataType::Utf8, true),
        Field::new("idempotency_key", DataType::Utf8, true),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(vectors),
            Arc::new(metadata_json),
            Arc::new(idempotency_key),
        ],
    )
    .expect("record batch")
}

async fn flight_do_put(
    endpoint: &str,
    token: Option<&str>,
    batch: RecordBatch,
) -> Result<Vec<PutResult>, FlightError> {
    let channel = tonic::transport::Endpoint::from_shared(endpoint.to_string())
        .expect("valid endpoint")
        .connect()
        .await
        .expect("connect flight endpoint");
    let mut client = FlightClient::new(channel);
    if let Some(token) = token {
        client
            .add_header("authorization", &format!("Bearer {token}"))
            .expect("set authorization metadata");
    }

    let stream = FlightDataEncoderBuilder::new().build(futures::stream::iter(vec![Ok(batch)]));
    let put_stream = client.do_put(stream).await?;
    put_stream.try_collect::<Vec<_>>().await
}

#[tokio::test]
async fn flight_ingest_roundtrip_and_idempotency_ack() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("flight_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let token = create_api_key(&data_dir).expect("create api key");
    let Some(http_port) = reserve_local_port() else {
        eprintln!("skipping flight e2e: localhost http bind unavailable");
        return;
    };
    let Some(flight_port) = reserve_local_port() else {
        eprintln!("skipping flight e2e: localhost flight bind unavailable");
        return;
    };

    let base_url = format!("http://127.0.0.1:{http_port}");
    let flight_endpoint = format!("http://127.0.0.1:{flight_port}");

    let mut server = start_server(&data_dir, http_port, flight_port)
        .await
        .expect("spawn server");

    if let Err(e) = wait_for_ready(&base_url, &token).await {
        stop_server_with_logs(&mut server).await;
        panic!("ready check failed: {e}");
    }
    if let Err(e) = wait_for_flight_ready(&flight_endpoint).await {
        stop_server_with_logs(&mut server).await;
        panic!("flight readiness check failed: {e}");
    }

    let rows = vec![
        FlightRow {
            vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata_json: Some(
                r#"{"source_file":"a.wav","start_time_ms":10,"duration_ms":15,"bpm":120.0,"tags":["flight","drums"]}"#
                    .to_string(),
            ),
            idempotency_key: Some("flight-k1".to_string()),
        },
        FlightRow {
            vector: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata_json: Some(
                r#"{"source_file":"b.wav","start_time_ms":20,"duration_ms":12,"bpm":121.0,"tags":["flight","bass"]}"#
                    .to_string(),
            ),
            idempotency_key: Some("flight-k2".to_string()),
        },
        FlightRow {
            vector: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata_json: None,
            idempotency_key: Some("flight-k3".to_string()),
        },
        FlightRow {
            vector: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            metadata_json: None,
            idempotency_key: Some("flight-k4".to_string()),
        },
    ];

    let batch = build_batch(&rows, DIM);
    let response = flight_do_put(&flight_endpoint, Some(&token), batch)
        .await
        .expect("first flight do_put");
    assert_eq!(response.len(), 1, "expected single put ack response");
    let ack: serde_json::Value =
        serde_json::from_slice(response[0].app_metadata.as_ref()).expect("ack metadata json");
    assert_eq!(ack["accepted"].as_u64(), Some(rows.len() as u64));
    assert_eq!(ack["created"].as_u64(), Some(rows.len() as u64));

    let second_batch = build_batch(&rows, DIM);
    let second_response = flight_do_put(&flight_endpoint, Some(&token), second_batch)
        .await
        .expect("second flight do_put");
    let second_ack: serde_json::Value =
        serde_json::from_slice(second_response[0].app_metadata.as_ref()).expect("second ack json");
    assert_eq!(second_ack["accepted"].as_u64(), Some(rows.len() as u64));
    assert_eq!(second_ack["created"].as_u64(), Some(0));

    let client = reqwest::Client::new();
    let stats_resp = client
        .get(format!("{}/v2/admin/stats", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("stats request");
    assert_eq!(stats_resp.status(), StatusCode::OK);
    let stats: serde_json::Value = stats_resp.json().await.expect("stats json");
    assert_eq!(
        stats["data"]["total_vectors"].as_u64(),
        Some(rows.len() as u64)
    );

    for (idx, row) in rows.iter().enumerate() {
        let query = serde_json::json!({
            "vector": row.vector,
            "k": 10,
            "ef": 64,
            "include_metadata": false,
            "search_tier": "active"
        });
        let query_resp = client
            .post(format!("{}/v2/query", base_url))
            .bearer_auth(&token)
            .json(&query)
            .send()
            .await
            .expect("query request");
        assert_eq!(
            query_resp.status(),
            StatusCode::OK,
            "query failed for row {idx}"
        );
        let parsed: serde_json::Value = query_resp.json().await.expect("query json");
        let results = parsed["data"]["results"]
            .as_array()
            .expect("query results array");
        assert!(
            !results.is_empty(),
            "query returned no results for row {idx}"
        );
    }

    stop_server(&mut server).await;
}

#[tokio::test]
async fn flight_ingest_requires_auth_and_rejects_dim_mismatch() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("flight_auth_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let token = create_api_key(&data_dir).expect("create api key");
    let Some(http_port) = reserve_local_port() else {
        eprintln!("skipping flight auth e2e: localhost http bind unavailable");
        return;
    };
    let Some(flight_port) = reserve_local_port() else {
        eprintln!("skipping flight auth e2e: localhost flight bind unavailable");
        return;
    };

    let base_url = format!("http://127.0.0.1:{http_port}");
    let flight_endpoint = format!("http://127.0.0.1:{flight_port}");

    let mut server = start_server(&data_dir, http_port, flight_port)
        .await
        .expect("spawn server");

    if let Err(e) = wait_for_ready(&base_url, &token).await {
        stop_server_with_logs(&mut server).await;
        panic!("ready check failed: {e}");
    }
    if let Err(e) = wait_for_flight_ready(&flight_endpoint).await {
        stop_server_with_logs(&mut server).await;
        panic!("flight readiness check failed: {e}");
    }

    let rows = vec![FlightRow {
        vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        metadata_json: None,
        idempotency_key: Some("auth-k1".to_string()),
    }];

    let unauth_batch = build_batch(&rows, DIM);
    let unauth_err = flight_do_put(&flight_endpoint, None, unauth_batch)
        .await
        .expect_err("expected unauthenticated flight ingest failure");
    match unauth_err {
        FlightError::Tonic(status) => {
            assert_eq!(status.code(), tonic::Code::Unauthenticated);
        }
        other => panic!("expected tonic unauthenticated error, got {other}"),
    }

    let bad_dim_batch = build_batch(&rows, DIM + 1);
    let bad_dim_err = flight_do_put(&flight_endpoint, Some(&token), bad_dim_batch)
        .await
        .expect_err("expected invalid-argument for dim mismatch");
    match bad_dim_err {
        FlightError::Tonic(status) => {
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
            assert!(
                status.message().contains("dimension mismatch"),
                "unexpected error message: {}",
                status.message()
            );
        }
        other => panic!("expected tonic invalid-argument error, got {other}"),
    }

    stop_server(&mut server).await;
}
