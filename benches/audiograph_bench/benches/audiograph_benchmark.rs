use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// Import directly from the public modules
use audiograph::buffer::{AudioBuffer, FrameSize, MultiChannelBuffer};
use audiograph::processor::{ProcessingContext, Processor};

struct BenchProcessor {
    increment: f32,
}

impl BenchProcessor {
    fn new(increment: f32) -> Self {
        Self { increment }
    }
}

// Empty processor for measuring pure graph overhead
struct EmptyProcessor;

impl EmptyProcessor {
    fn new() -> Self {
        Self
    }
}

impl Processor<f32> for BenchProcessor {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        for channel in context.channel_layout.iter() {
            let input_channel = context.input_buffer.channel(channel).unwrap();
            let output_channel = context.output_buffer.channel_mut(channel).unwrap();
            for (out_sample, in_sample) in output_channel.iter_mut().zip(input_channel.iter()) {
                *out_sample += in_sample + self.increment;
            }
        }
    }
}

impl Processor<f32> for EmptyProcessor {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
    }
}

fn create_diamond_graph(layers: usize, buffer_size: usize, use_empty: bool) -> (audiograph::DspGraph<f32>, usize) {
    use audiograph::*;

    assert!(layers % 2 == 1, "Layers must be odd");
    assert!(layers >= 3, "Need at least 3 layers");

    let middle_layer = layers / 2;
    let increment = 0.1;

    let mut total_edges = 0;
    let mut layer_sizes = Vec::new();
    for layer in 0..layers {
        if layer == 0 {
            layer_sizes.push(2);
            total_edges += 2;
        } else if layer <= middle_layer {
            let nodes_in_layer = 2_usize.pow(layer as u32 + 1);
            layer_sizes.push(nodes_in_layer);
            total_edges += nodes_in_layer * 2;
        } else {
            let nodes_in_layer = 2_usize.pow((layers - layer) as u32);
            layer_sizes.push(nodes_in_layer);
            total_edges += nodes_in_layer;
        }
    }

    let mut graph = DspGraph::new(2, FrameSize(buffer_size), Some(total_edges));

    let mut all_layers = Vec::new();
    for layer in 0..layers {
        let nodes_in_layer = layer_sizes[layer];

        let mut layer_nodes = Vec::new();
        for _i in 0..nodes_in_layer {
            let processor_node = if use_empty {
                graph
                    .add_processor(
                        EmptyProcessor::new(),
                        MultiChannelBuffer::new(2, FrameSize(buffer_size)),
                    )
                    .unwrap()
            } else {
                graph
                    .add_processor(
                        BenchProcessor::new(increment),
                        MultiChannelBuffer::new(2, FrameSize(buffer_size)),
                    )
                    .unwrap()
            };
            layer_nodes.push(processor_node);
        }
        all_layers.push(layer_nodes);
    }

    for node in &all_layers[0] {
        graph
            .connect(GraphNode::Input, (*node).into(), None)
            .unwrap();
    }

    for layer in 0..layers - 1 {
        let current_layer = &all_layers[layer];
        let next_layer = &all_layers[layer + 1];

        if layer < middle_layer {
            // Expanding phase
            for (i, &current_node) in current_layer.iter().enumerate() {
                let next_start = i * 2;
                graph
                    .connect(current_node.into(), next_layer[next_start].into(), None)
                    .unwrap();
                graph
                    .connect(current_node.into(), next_layer[next_start + 1].into(), None)
                    .unwrap();
            }
        } else {
            // Contracting phase
            for (i, &next_node) in next_layer.iter().enumerate() {
                let current_start = i * 2;
                graph
                    .connect(current_layer[current_start].into(), next_node.into(), None)
                    .unwrap();
                graph
                    .connect(
                        current_layer[current_start + 1].into(),
                        next_node.into(),
                        None,
                    )
                    .unwrap();
            }
        }
    }

    // Connect last layer to output
    let last_layer = &all_layers[layers - 1];
    for node in last_layer {
        graph
            .connect((*node).into(), GraphNode::Output, None)
            .unwrap();
    }

    (graph, total_edges)
}

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
