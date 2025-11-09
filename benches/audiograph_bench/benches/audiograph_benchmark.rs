use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// Import directly from the public modules
use audiograph::buffer::{FrameSize, MultiChannelBuffer};
use audiograph::processor::{ProcessingContext, Processor};

struct BenchProcessor;

impl Processor<f32> for BenchProcessor {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        for channel in context.channel_layout.iter() {
            let output_channel = context.output_buffer.channel_mut(channel).unwrap();
            for sample in output_channel.iter_mut() {
                *sample = 0.5;
            }
        }
    }
}

fn create_small_graph() -> (
    audiograph::DspGraph<f32>,
    MultiChannelBuffer<f32>,
    MultiChannelBuffer<f32>,
) {
    use audiograph::*;

    let mut graph = DspGraph::new(2, FrameSize(64), None);
    let processor_node = graph
        .add_processor(BenchProcessor, MultiChannelBuffer::new(2, FrameSize(64)))
        .unwrap();

    graph
        .connect(GraphNode::Input, processor_node.into(), None)
        .unwrap();
    graph
        .connect(processor_node.into(), GraphNode::Output, None)
        .unwrap();

    let input = MultiChannelBuffer::new(2, FrameSize(64));
    let output = MultiChannelBuffer::new(2, FrameSize(64));

    (graph, input, output)
}

fn bench_audiograph_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("audiograph_performance");

    // Test different buffer sizes (advantage over DASP's fixed 64)
    for &buffer_size in &[64, 256, 1024] {
        use audiograph::*;

        let mut graph = DspGraph::new(2, FrameSize(buffer_size), Some(10));
        let processor_node = graph
            .add_processor(
                BenchProcessor,
                MultiChannelBuffer::new(2, FrameSize(buffer_size)),
            )
            .unwrap();

        graph
            .connect(GraphNode::Input, processor_node.into(), None)
            .unwrap();
        graph
            .connect(processor_node.into(), GraphNode::Output, None)
            .unwrap();

        let input = MultiChannelBuffer::new(2, FrameSize(buffer_size));
        let mut output = MultiChannelBuffer::new(2, FrameSize(buffer_size));

        group.bench_function(&format!("buffer_size_{}", buffer_size), |b| {
            b.iter(|| {
                graph.process(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(FrameSize(buffer_size)),
                );
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_audiograph_performance);
criterion_main!(benches);
