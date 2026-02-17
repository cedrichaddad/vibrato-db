//! HNSW Benchmarks
//!
//! Run with: cargo bench --bench hnsw

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::Arc;
use std::time::Duration;
use vibrato_db::hnsw::HNSW;
use vibrato_db::simd::l2_normalized;

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    l2_normalized(&v)
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");

    // Insert is slow (O(n log n)), so use fewer samples
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Only test smaller sizes for insert (each iteration rebuilds index)
    for num_vectors in [100, 500, 1000] {
        let mut rng = StdRng::seed_from_u64(1000 + num_vectors as u64);
        let vectors: Vec<_> = (0..num_vectors)
            .map(|_| random_vector(128, &mut rng))
            .collect();
        let vectors = Arc::new(vectors);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            &num_vectors,
            |b, &n| {
                b.iter(|| {
                    let vectors = vectors.clone();
                    let mut hnsw = HNSW::new_with_accessor_and_seed(
                        16,
                        100,
                        move |id, sink| sink(&vectors[id]),
                        42,
                    );
                    for i in 0..n {
                        hnsw.insert(i);
                    }
                    black_box(hnsw.nodes.len())
                })
            },
        );
    }

    group.finish();
}

fn bench_search(c: &mut Criterion) {
    // Build index once (use smaller index for faster setup)
    let num_vectors = 5000;
    let mut rng = StdRng::seed_from_u64(4242);
    let vectors: Vec<_> = (0..num_vectors)
        .map(|_| random_vector(128, &mut rng))
        .collect();
    let vectors = Arc::new(vectors);

    let vectors_clone = vectors.clone();
    let mut hnsw =
        HNSW::new_with_accessor_and_seed(16, 100, move |id, sink| sink(&vectors_clone[id]), 42);
    for i in 0..num_vectors {
        hnsw.insert(i);
    }

    let mut group = c.benchmark_group("hnsw_search");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    for ef in [20, 50, 100] {
        let mut query_rng = StdRng::seed_from_u64(ef as u64 + 5000);
        let query = random_vector(128, &mut query_rng);

        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            b.iter(|| black_box(hnsw.search(&query, 10, ef)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_insert, bench_search);
criterion_main!(benches);
