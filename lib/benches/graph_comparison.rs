use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[derive(Debug, Clone)]
enum GraphSize {
    Small,
    Medium,
    Large,
}

// fn create_audiograph_chain(size: &GraphSize) {}
// fn create_dasp_chain(size: &GraphSize) {}

fn bench_all_combinations(c: &mut Criterion) {
    let mut group = c.benchmark_group("processing_performance");

    let buffer_sizes = [64, 512, 2048];
    let graph_sizes = [GraphSize::Small, GraphSize::Medium, GraphSize::Large];

    for &buffer_size in &buffer_sizes {
        for graph_size in &graph_sizes {
            let param_name = format!("buffer_size{}_{:?}", buffer_size, graph_size);

            group.bench_with_input(
                BenchmarkId::new("audiograph", &param_name),
                &(buffer_size, graph_size),
                |b, &(buf_size, graph_sz)| b.iter(|| {}),
            );

            group.bench_with_input(
                BenchmarkId::new("dasp_graph", &param_name),
                &(buffer_size, graph_size),
                |b, &(buf_size, graph_sz)| b.iter(|| {}),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_all_combinations);
criterion_main!(benches);
