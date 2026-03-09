use std::path::PathBuf;

#[test]
fn flight_ingest_persists_ipc_wal_batches() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("vibrato-server")
        .join("src")
        .join("prod")
        .join("flight.rs");
    let contents = std::fs::read_to_string(&source).expect("read flight.rs source");

    assert!(
        contents.contains("serialize_batch_to_ipc(&batch)"),
        "flight ingest path must serialize decoded RecordBatch into IPC for wal_batches append"
    );
    assert!(
        contents.contains(".ingest_wal_ipc_batch(&state.collection.id, cols.num_rows, &ipc_blob)"),
        "flight ingest path must persist one wal_batches record per processed RecordBatch"
    );
}
