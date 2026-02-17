//! SIMD Benchmarks
//!
//! Run with: cargo bench --bench simd

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::Rng;
use vibrato_db::simd::{dot_product, l2_distance_squared};

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()
}

fn bench_dot_product(c: &mut Criterion) {
    let dims = [64, 128, 256, 512];

    let mut group = c.benchmark_group("dot_product");

    for dim in dims {
        group.throughput(Throughput::Elements(dim as u64));

        let a = random_vector(dim);
        let b = random_vector(dim);

        group.bench_function(format!("dim_{}", dim), |bencher| {
            bencher.iter(|| dot_product(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_l2_distance(c: &mut Criterion) {
    let dims = [64, 128, 256, 512];

    let mut group = c.benchmark_group("l2_distance_squared");

    for dim in dims {
        group.throughput(Throughput::Elements(dim as u64));

        let a = random_vector(dim);
        let b = random_vector(dim);

        group.bench_function(format!("dim_{}", dim), |bencher| {
            bencher.iter(|| l2_distance_squared(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dot_product, bench_l2_distance);
criterion_main!(benches);
