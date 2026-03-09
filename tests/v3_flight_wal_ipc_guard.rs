use std::path::PathBuf;

#[test]
fn flight_ingest_does_not_write_wal_batches_directly() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("vibrato-server")
        .join("src")
        .join("prod")
        .join("flight.rs");
    let contents = std::fs::read_to_string(&source).expect("read flight.rs source");

    assert!(
        !contents.contains("serialize_batch_to_ipc(&batch)"),
        "flight ingest path must not serialize wal IPC blobs; this belongs to engine ingest path"
    );
    assert!(
        !contents.contains(".ingest_wal_ipc_batch("),
        "flight ingest path must not append wal_batches directly; engine owns dedupe/id assignment/wal append"
    );

    let engine_source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("vibrato-server")
        .join("src")
        .join("prod")
        .join("engine.rs");
    let engine_contents = std::fs::read_to_string(&engine_source).expect("read engine.rs source");
    assert!(
        engine_contents.contains("append_wal_ipc_batch_for_accepted_rows(&accepted_rows)"),
        "engine ingest path must append wal_batches from accepted deduplicated rows"
    );
}
