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
        contents.contains("ingest_batch_owned(entries)"),
        "flight ingest path must use owned batch handoff to avoid extra row-buffer cloning"
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
