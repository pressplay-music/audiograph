use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use audiograph::buffer::{AudioBuffer, FrameSize, MultiChannelBuffer};
use audiograph_bench::{create_diamond_graph, BenchProcessor, EmptyProcessor};

fn bench_audiograph_performance(c: &mut Criterion) {
    // Graph overhead only (EmptyProcessor)
    let mut overhead_group = c.benchmark_group("audiograph_overhead");
    for &layers in &[3, 5, 7, 9] {
        let buffer_size = 64;
        let (mut graph, total_edges) = create_diamond_graph(layers, buffer_size, true); // use_empty = true
        let input = MultiChannelBuffer::new(2, FrameSize(buffer_size));
        let mut output = MultiChannelBuffer::new(2, FrameSize(buffer_size));

        overhead_group.bench_function(&format!("graph_{}", total_edges), |b| {
            b.iter(|| {
                graph.process(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(FrameSize(buffer_size)),
                );
            })
        });
    }
    overhead_group.finish();

    // Full processing (BenchProcessor)
    let mut full_group = c.benchmark_group("audiograph_full");
    for &layers in &[3, 5, 7, 9] {
        let buffer_size = 64;
        let increment = 0.1;
        let (mut graph, total_edges) = create_diamond_graph(layers, buffer_size, false); // use_empty = false
        let input = MultiChannelBuffer::new(2, FrameSize(buffer_size));
        let mut output = MultiChannelBuffer::new(2, FrameSize(buffer_size));

        // Check output value before benchmarking
        graph.process(&input, &mut output, FrameSize(buffer_size));
        println!(
            "Audiograph {} layers -> output[0][0] = {}",
            layers,
            output.channel(0).unwrap()[0]
        );

        full_group.bench_function(&format!("graph_{}", total_edges), |b| {
            b.iter(|| {
                graph.process(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(FrameSize(buffer_size)),
                );
            })
        });
    }
    full_group.finish();
}

criterion_group!(benches, bench_audiograph_performance);
criterion_main!(benches);
