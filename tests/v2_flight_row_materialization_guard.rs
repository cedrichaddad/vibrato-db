use std::path::PathBuf;

#[test]
fn flight_ingest_path_uses_owned_batch_handoff() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("vibrato-server")
        .join("src")
        .join("prod")
        .join("flight.rs");
    let contents = std::fs::read_to_string(&source).expect("read flight.rs source");

    assert!(
        contents.contains("ingest_flight_batch_streaming(&state_bg, batch, dim)"),
        "flight ingest path must stream/decode batches in chunks instead of materializing all rows up front"
    );
    assert!(
        contents.contains("ingest_batch_owned(std::mem::take(&mut chunk))"),
        "flight ingest path must hand off owned chunk buffers to avoid extra row-buffer cloning"
    );
    assert!(
        !contents.contains("ingest_batch(&entries)"),
        "flight ingest path regressed to borrowed-slice ingest and reintroduced row-buffer clone"
    );
    assert!(
        !contents.contains("entries.to_vec()"),
        "flight ingest path should not clone entry buffers with entries.to_vec()"
    );
}
