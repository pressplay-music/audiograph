use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// DASP imports - uses petgraph 0.5 internally
use dasp_graph::node::Node;
use dasp_graph::{Buffer, NodeData};

struct BenchProcessor {
    increment: f32,
}

impl BenchProcessor {
    fn new(increment: f32) -> Self {
        Self { increment }
    }
}

enum GraphNode {
    Source,
    Processor(BenchProcessor),
    EmptyProcessor, // For measuring graph overhead
    Output,
}

impl Node for GraphNode {
    fn process(&mut self, inputs: &[dasp_graph::Input], outputs: &mut [Buffer]) {
        match self {
            GraphNode::Source => {
                for output in outputs.iter_mut() {
                    for sample in output.iter_mut() {
                        *sample = 0.0;
                    }
                }
            }
            GraphNode::Processor(processor) => {
                processor.process(inputs, outputs);
            }
            GraphNode::EmptyProcessor => {
                // Empty processor - only does summing (DASP's node responsibility)
                for output in outputs.iter_mut() {
                    for sample in output.iter_mut() {
                        *sample = 0.0;
                    }
                }
                
                for input in inputs.iter() {
                    for (output_buffer, input_buffer) in outputs.iter_mut().zip(input.buffers().iter()) {
                        for (out_sample, in_sample) in output_buffer.iter_mut().zip(input_buffer.iter()) {
                            *out_sample += in_sample; // Only summing, no increment
                        }
                    }
                }
            }
            GraphNode::Output => {
                for output in outputs.iter_mut() {
                    for sample in output.iter_mut() {
                        *sample = 0.0;
                    }
                }

                for input in inputs.iter() {
                    for (output_buffer, input_buffer) in
                        outputs.iter_mut().zip(input.buffers().iter())
                    {
                        for (out_sample, in_sample) in
                            output_buffer.iter_mut().zip(input_buffer.iter())
                        {
                            *out_sample += in_sample;
                        }
                    }
                }
            }
        }
    }
}

impl Node for BenchProcessor {
    fn process(&mut self, inputs: &[dasp_graph::Input], outputs: &mut [Buffer]) {
        for output in outputs.iter_mut() {
            for sample in output.iter_mut() {
                *sample = 0.0;
            }
        }

        // Sum all inputs into outputs (node's responsibility in DASP)
        for input in inputs.iter() {
            for (output_buffer, input_buffer) in outputs.iter_mut().zip(input.buffers().iter()) {
                for (out_sample, in_sample) in output_buffer.iter_mut().zip(input_buffer.iter()) {
                    *out_sample += in_sample;
                }
            }
        }

        // Add increment
        for output in outputs.iter_mut() {
            for sample in output.iter_mut() {
                *sample += self.increment;
            }
        }
    }
}

type DaspGraph = petgraph::graph::DiGraph<NodeData<GraphNode>, ()>;
type DaspProcessor = dasp_graph::Processor<DaspGraph>;

fn create_dasp_diamond_graph(
    layers: usize,
    use_empty: bool,
) -> (DaspGraph, DaspProcessor, petgraph::graph::NodeIndex, usize) {
    assert!(layers % 2 == 1, "Layers must be odd");
    assert!(layers >= 3, "Need at least 3 layers");

    let middle_layer = layers / 2;
    let increment = 0.1;

    // Use identical edge counting logic as audiograph
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

    let total_nodes: usize = layer_sizes.iter().sum::<usize>() + 2; // +1 for source, +1 for output
    // Note: Don't add extra edges to total_edges - use audiograph's counting method

    let mut graph = DaspGraph::with_capacity(total_nodes, total_edges);
    let processor = DaspProcessor::with_capacity(total_nodes);

    let source_node_data = NodeData::new2(GraphNode::Source);
    let source_node = graph.add_node(source_node_data);

    let output_node_data = NodeData::new2(GraphNode::Output);
    let output_node = graph.add_node(output_node_data);

    // Create layers of nodes
    let mut all_layers = Vec::new();
    for layer in 0..layers {
        let nodes_in_layer = layer_sizes[layer];

        let mut layer_nodes = Vec::new();
        for _i in 0..nodes_in_layer {
            let graph_node = if use_empty {
                GraphNode::EmptyProcessor
            } else {
                GraphNode::Processor(BenchProcessor::new(increment))
            };
            let node_data = NodeData::new2(graph_node);
            let node_id = graph.add_node(node_data);
            layer_nodes.push(node_id);
        }
        all_layers.push(layer_nodes);
    }

    // Connect source to first layer (equivalent to Input → first layer)
    for node in &all_layers[0] {
        graph.add_edge(source_node, *node, ());
    }

    // Connect between layers
    for layer in 0..layers - 1 {
        let current_layer = &all_layers[layer];
        let next_layer = &all_layers[layer + 1];

        if layer < middle_layer {
            // Expanding phase: each node connects to 2 nodes in next layer
            for (i, &current_node) in current_layer.iter().enumerate() {
                let next_start = i * 2;
                graph.add_edge(current_node, next_layer[next_start], ());
                graph.add_edge(current_node, next_layer[next_start + 1], ());
            }
        } else {
            // Contracting phase: 2 nodes connect to each node in next layer
            for (i, &next_node) in next_layer.iter().enumerate() {
                let current_start = i * 2;
                graph.add_edge(current_layer[current_start], next_node, ());
                graph.add_edge(current_layer[current_start + 1], next_node, ());
            }
        }
    }

    // Connect last layer to output node (equivalent to last layer → Output)
    for &last_node in &all_layers[layers - 1] {
        graph.add_edge(last_node, output_node, ());
    }

    // Return output node as entry point
    (graph, processor, output_node, total_edges)
}

fn bench_dasp_performance(c: &mut Criterion) {
    // Graph overhead only (EmptyProcessor)
    let mut overhead_group = c.benchmark_group("dasp_overhead");
    for &layers in &[3, 5, 7, 9] {
        let (mut dasp_graph, mut dasp_processor, node_id, total_edges) =
            create_dasp_diamond_graph(layers, true);

        overhead_group.bench_function(&format!("graph_{}", total_edges), |b| {
            b.iter(|| {
                dasp_processor.process(black_box(&mut dasp_graph), black_box(node_id));
            })
        });
    }
    overhead_group.finish();

    // Full processing (BenchProcessor)
    let mut full_group = c.benchmark_group("dasp_full");
    for &layers in &[3, 5, 7, 9] {
        let (mut dasp_graph, mut dasp_processor, node_id, total_edges) =
            create_dasp_diamond_graph(layers, false); // use_empty = false

        // Check output value before benchmarking
        dasp_processor.process(&mut dasp_graph, node_id);
        let output_buffer = dasp_graph
            .node_weight(node_id)
            .unwrap()
            .buffers
            .first()
            .unwrap();
        println!(
            "DASP {} layers -> output[0][0] = {}",
            layers, output_buffer[0]
        );

        full_group.bench_function(&format!("graph_{}", total_edges), |b| {
            b.iter(|| {
                dasp_processor.process(black_box(&mut dasp_graph), black_box(node_id));
            })
        });
    }
    full_group.finish();
}

criterion_group!(benches, bench_dasp_performance);
criterion_main!(benches);
