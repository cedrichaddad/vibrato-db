use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vibrato_core::metadata::VectorMetadata;
use vibrato_core::simd::dot_product;
use vibrato_db::prod::model::{QueryFilter, QueryRequestV2, SearchTier};
use vibrato_db::prod::{bootstrap_data_dirs, ProductionConfig, ProductionState, SqliteCatalog};

const DIM: usize = 16;
const TAG_POOL: &[&str] = &[
    "drums", "kick", "snare", "hihat", "bass", "pad", "vocal", "fx",
];

#[derive(Clone, Debug)]
struct Op {
    roll: u8,
    a: u16,
    b: u16,
    c: u16,
    d: u16,
}

#[derive(Clone, Debug)]
struct ModelRow {
    id: usize,
    vector: Vec<f32>,
    metadata: VectorMetadata,
}

#[derive(Default, Clone, Debug)]
struct ReferenceModel {
    rows: Vec<ModelRow>,
    by_idempotency: HashMap<String, usize>,
    next_id: usize,
}

impl ReferenceModel {
    fn ingest(
        &mut self,
        vector: Vec<f32>,
        metadata: VectorMetadata,
        idempotency_key: Option<String>,
    ) -> (usize, bool) {
        if let Some(key) = idempotency_key.as_ref() {
            if let Some(existing) = self.by_idempotency.get(key) {
                return (*existing, false);
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        if let Some(key) = idempotency_key {
            self.by_idempotency.insert(key, id);
        }
        self.rows.push(ModelRow {
            id,
            vector,
            metadata,
        });
        (id, true)
    }

    fn by_id(&self, id: usize) -> Option<&ModelRow> {
        self.rows.iter().find(|row| row.id == id)
    }

    fn total(&self) -> usize {
        self.rows.len()
    }
}

fn normalized_vector(seed: u64, nonce: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed ^ nonce.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        v.push(rng.gen::<f32>() * 2.0 - 1.0);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn normalized_metadata(
    tags: Vec<String>,
    bpm: f32,
    source: String,
    start_ms: u32,
) -> VectorMetadata {
    let mut out_tags = tags
        .into_iter()
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>();
    out_tags.sort();
    out_tags.dedup();
    VectorMetadata {
        source_file: source,
        start_time_ms: start_ms,
        duration_ms: 80,
        bpm,
        tags: out_tags,
    }
}

fn metadata_from_op(step: usize, op: &Op) -> VectorMetadata {
    let tag_a = TAG_POOL[(op.a as usize) % TAG_POOL.len()].to_string();
    let tag_b = TAG_POOL[(op.b as usize) % TAG_POOL.len()].to_string();
    let mut tags = vec![tag_a];
    if op.c % 2 == 0 {
        tags.push(tag_b);
    }
    normalized_metadata(
        tags,
        70.0 + ((op.d % 160) as f32) * 0.5,
        format!("prop_{step}.wav"),
        (step as u32).saturating_mul(5),
    )
}

fn idempotency_for_op(step: usize, op: &Op) -> Option<String> {
    match op.c % 5 {
        0 => None,
        1 | 2 => Some(format!("shared-{}", op.a % 32)),
        _ => Some(format!("unique-{}-{}", step, op.b)),
    }
}

fn make_filter_from_op(row: &ModelRow, op: &Op) -> Option<QueryFilter> {
    match op.roll % 5 {
        0 => None,
        1 => Some(QueryFilter {
            tags_all: vec![row
                .metadata
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "drums".to_string())],
            ..Default::default()
        }),
        2 => Some(QueryFilter {
            tags_any: vec![
                row.metadata
                    .tags
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "drums".to_string()),
                "missing-tag".to_string(),
            ],
            ..Default::default()
        }),
        3 => Some(QueryFilter {
            bpm_gte: Some(row.metadata.bpm - 0.01),
            bpm_lte: Some(row.metadata.bpm + 0.01),
            ..Default::default()
        }),
        _ => Some(QueryFilter {
            tags_all: vec![row
                .metadata
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "drums".to_string())],
            bpm_gte: Some(row.metadata.bpm - 0.01),
            bpm_lte: Some(row.metadata.bpm + 0.01),
            ..Default::default()
        }),
    }
}

fn filter_matches(metadata: &VectorMetadata, filter: Option<&QueryFilter>) -> bool {
    let Some(filter) = filter else {
        return true;
    };

    if !filter.tags_all.is_empty()
        && !filter.tags_all.iter().all(|t| {
            let norm = t.trim().to_ascii_lowercase();
            metadata.tags.iter().any(|m| m == &norm)
        })
    {
        return false;
    }

    if !filter.tags_any.is_empty()
        && !filter.tags_any.iter().any(|t| {
            let norm = t.trim().to_ascii_lowercase();
            metadata.tags.iter().any(|m| m == &norm)
        })
    {
        return false;
    }

    if let Some(min_bpm) = filter.bpm_gte {
        if metadata.bpm < min_bpm {
            return false;
        }
    }
    if let Some(max_bpm) = filter.bpm_lte {
        if metadata.bpm > max_bpm {
            return false;
        }
    }
    true
}

fn expected_best_id(
    model: &ReferenceModel,
    query: &[f32],
    filter: Option<&QueryFilter>,
) -> Option<usize> {
    model
        .rows
        .iter()
        .filter(|row| filter_matches(&row.metadata, filter))
        .map(|row| (row.id, dot_product(query, &row.vector)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
}

fn assert_stats_total(state: &Arc<ProductionState>, expected: usize, seed: u64, step: usize) {
    let stats = state.stats().expect("stats");
    assert_eq!(
        stats.total_vectors, expected,
        "stats.total_vectors mismatch seed={} step={} expected={} actual={}",
        seed, step, expected, stats.total_vectors
    );
}

fn metadata_eq(left: &VectorMetadata, right: &VectorMetadata) -> bool {
    left.source_file == right.source_file
        && left.start_time_ms == right.start_time_ms
        && left.duration_ms == right.duration_ms
        && (left.bpm - right.bpm).abs() < 1e-6
        && left.tags == right.tags
}

fn new_state() -> Arc<ProductionState> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let data_dir: PathBuf = std::env::temp_dir().join(format!("vibrato_prop_model_{nonce}"));
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let mut config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), DIM);
    config.checkpoint_interval = Duration::from_secs(3600);
    config.compaction_interval = Duration::from_secs(3600);
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = Arc::new(SqliteCatalog::open(&config.catalog_path()).expect("open catalog"));
    let state = ProductionState::initialize(config, catalog).expect("initialize state");
    state.set_ready(true, "property-test");
    state
}

fn generate_ops(seed: u64, count: usize) -> Vec<Op> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA11C_5EED_99AA_0042);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(Op {
            roll: rng.gen_range(0..100),
            a: rng.gen(),
            b: rng.gen(),
            c: rng.gen(),
            d: rng.gen(),
        });
    }
    out
}

fn run_reference_seed(seed: u64, ops: &[Op]) {
    let state = new_state();
    let mut model = ReferenceModel::default();
    let mut checkpoints = 0usize;
    let mut compactions = 0usize;

    for (step, op) in ops.iter().enumerate() {
        if op.roll < 45 {
            let vector = normalized_vector(seed, (step as u64) ^ (op.a as u64));
            let metadata = metadata_from_op(step, op);
            let idempotency = idempotency_for_op(step, op);

            let actual = state
                .ingest_vector(&vector, &metadata, idempotency.as_deref())
                .expect("engine ingest");
            let expected = model.ingest(vector, metadata, idempotency);
            assert_eq!(
                actual, expected,
                "ingest mismatch seed={} step={} op={:?}",
                seed, step, op
            );
        } else if op.roll < 65 {
            let batch_size = ((op.a as usize) % 6) + 1;
            let mut entries = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let nonce = ((step as u64) << 16) ^ ((i as u64) << 8) ^ (op.b as u64);
                let vector = normalized_vector(seed, nonce);
                let metadata = metadata_from_op(step * 16 + i, op);
                let idempotency = if i % 3 == 0 {
                    Some(format!("batch-shared-{}", op.c % 24))
                } else {
                    idempotency_for_op(step * 16 + i, op)
                };
                entries.push((vector, metadata, idempotency));
            }

            let actual = state.ingest_batch(&entries).expect("engine batch ingest");
            let expected = entries
                .iter()
                .map(|(v, m, k)| model.ingest(v.clone(), m.clone(), k.clone()))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "batch mismatch seed={} step={} op={:?}",
                seed, step, op
            );
        } else if op.roll < 90 {
            if model.total() == 0 {
                continue;
            }
            let idx = (op.a as usize) % model.total();
            let target = model.rows[idx].clone();
            let filter = make_filter_from_op(&target, op);
            let expected_best = expected_best_id(&model, &target.vector, filter.as_ref());
            let k = model.total().min(32).max(1);
            let request = QueryRequestV2 {
                vector: target.vector.clone(),
                k,
                ef: (k * 16).max(256),
                include_metadata: true,
                filter: filter.clone(),
                search_tier: SearchTier::All,
            };

            let response = state.query(&request).expect("query");
            for result in &response.results {
                let row = model
                    .by_id(result.id)
                    .expect("query id should exist in model");
                assert!(
                    filter_matches(&row.metadata, filter.as_ref()),
                    "query leaked id={} not matching filter={:?} seed={} step={} op={:?}",
                    result.id,
                    filter,
                    seed,
                    step,
                    op
                );
                assert_eq!(
                    result.metadata.is_some(),
                    true,
                    "missing metadata in query response seed={} step={} id={}",
                    seed,
                    step,
                    result.id
                );
                let got = result.metadata.as_ref().expect("metadata present");
                assert!(
                    metadata_eq(got, &row.metadata),
                    "metadata mismatch seed={} step={} id={} expected={:?} got={:?}",
                    seed,
                    step,
                    result.id,
                    row.metadata,
                    got
                );
            }

            if let Some(best_id) = expected_best {
                assert!(
                    response.results.iter().any(|r| r.id == best_id),
                    "expected best_id={} missing from query results seed={} step={} top={:?}",
                    best_id,
                    seed,
                    step,
                    response
                        .results
                        .iter()
                        .take(6)
                        .map(|r| r.id)
                        .collect::<Vec<_>>()
                );
            } else {
                assert!(
                    response.results.is_empty(),
                    "expected empty result set seed={} step={} op={:?}",
                    seed,
                    step,
                    op
                );
            }
        } else if op.roll < 95 {
            state.checkpoint_once().expect("checkpoint_once");
            checkpoints += 1;
        } else {
            state.compact_once().expect("compact_once");
            compactions += 1;
        }

        if step % 25 == 0 {
            assert_stats_total(&state, model.total(), seed, step);
        }
    }

    assert_stats_total(&state, model.total(), seed, ops.len());

    // Old and new vectors must remain queryable after compactions/checkpoints.
    let mut probe_ids = Vec::new();
    probe_ids.extend(model.rows.iter().take(6).map(|row| row.id));
    probe_ids.extend(model.rows.iter().rev().take(6).map(|row| row.id));
    probe_ids.sort_unstable();
    probe_ids.dedup();

    for id in probe_ids {
        let row = model.by_id(id).expect("probe id exists");
        let req = QueryRequestV2 {
            vector: row.vector.clone(),
            k: 20,
            ef: 512,
            include_metadata: false,
            filter: None,
            search_tier: SearchTier::All,
        };
        let resp = state.query(&req).expect("probe query");
        assert!(
            resp.results.iter().any(|r| r.id == id),
            "post-admin probe missing id={} seed={} checkpoints={} compactions={}",
            id,
            seed,
            checkpoints,
            compactions
        );
    }
}

#[test]
fn randomized_reference_model_matches_engine() {
    let seeds = std::env::var("VIBRATO_PROPTEST_SEEDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(64);
    let ops_per_seed = std::env::var("VIBRATO_PROPTEST_OPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(180)
        .max(10);

    for seed in 0..seeds {
        let ops = generate_ops(seed, ops_per_seed);
        run_reference_seed(seed, &ops);
    }
}
