use audiograph::*;

pub struct BenchProcessor {
    increment: f32,
}

impl BenchProcessor {
    fn new(increment: f32) -> Self {
        Self { increment }
    }
}

// Empty processor for measuring pure graph overhead
pub struct EmptyProcessor;

impl EmptyProcessor {
    fn new() -> Self {
        Self
    }
}

impl Processor<f32> for BenchProcessor {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        context.for_each_channel(|input_channel, output_channel| {
            for (out_sample, in_sample) in output_channel.iter_mut().zip(input_channel.iter()) {
                *out_sample += in_sample + self.increment;
            }
        });
    }
}

impl Processor<f32> for EmptyProcessor {
    fn process(&mut self, _context: &mut ProcessingContext<f32>) {}
}

pub fn create_diamond_graph(
    layers: usize,
    buffer_size: usize,
    use_empty: bool,
) -> (audiograph::DspGraph<f32>, usize) {
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
