use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphNode {
    Input,
    Output,
    Node(NodeIndex),
}

impl From<NodeIndex> for GraphNode {
    fn from(value: NodeIndex) -> Self {
        GraphNode::Node(value)
    }
}

pub trait AudioBuffer {
    fn clear(&mut self);
}

pub struct MultiChannelBuffer {
    pub data: Box<[f32]>,
}

impl MultiChannelBuffer {
    pub fn clear(&mut self) {
        // Clear all channels
    }

    pub fn add(&mut self, _other: &MultiChannelBuffer) {
        // Add other buffer to self
        println!("Adding buffer");
    }

    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size].into_boxed_slice(),
        }
    }
}

pub struct ProcessingContext<'a> {
    input_buffer: &'a MultiChannelBuffer,
    output_buffer: &'a mut MultiChannelBuffer,
    channel_layout: ChannelLayout,
}

#[derive(Clone)]
pub struct ChannelLayout {}

pub trait Processor {
    fn process(&mut self, context: &mut ProcessingContext);
}

pub struct PassThrough;

impl Processor for PassThrough {
    fn process(&mut self, context: &mut ProcessingContext) {
        context
            .output_buffer
            .data
            .copy_from_slice(&context.input_buffer.data);
    }
}

/// Represents an audio processing node
pub struct ProcessorNode {
    processor: Box<dyn Processor + Send>,
}

impl ProcessorNode {
    pub fn new<T: Processor + Send + 'static>(processor: T) -> Self {
        Self {
            processor: Box::new(processor),
        }
    }

    pub fn process(
        &mut self,
        input_buffer: &MultiChannelBuffer,
        output_buffer: &mut MultiChannelBuffer,
        channel_layout: ChannelLayout,
    ) {
        let mut context = ProcessingContext {
            input_buffer,
            output_buffer,
            channel_layout,
        };
        self.processor.process(&mut context);
    }
}

pub struct ProcessorChannel {
    pub channel_layout: ChannelLayout,
}

impl ProcessorChannel {
    pub fn get_layout(&self) -> ChannelLayout {
        self.channel_layout.clone()
    }
}

pub struct DspGraph {
    graph: DiGraph<ProcessorNode, ProcessorChannel>,
    topo_order: Vec<NodeIndex>, // Precomputed processing order
    buffers: Vec<MultiChannelBuffer>,
    buffer_map: HashMap<NodeIndex, usize>,
    input_edges: Vec<(NodeIndex, ChannelLayout)>, // Nodes that receive from input
    output_edges: Vec<(NodeIndex, ChannelLayout)>, // Nodes that send to output
    summing_buffer: MultiChannelBuffer,
}

impl DspGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            topo_order: Vec::new(),
            buffers: Vec::new(),
            buffer_map: HashMap::new(),
            input_edges: Vec::new(),
            output_edges: Vec::new(),
            summing_buffer: MultiChannelBuffer {
                data: vec![0.0; 256].into_boxed_slice(),
            },
        }
    }

    pub fn add_processor<T: Processor + Send + 'static>(
        &mut self,
        processor: T,
        output_buffer: MultiChannelBuffer,
    ) -> NodeIndex {
        let node_index = self.graph.add_node(ProcessorNode::new(processor));

        // Store buffer and map NodeIndex -> Vec index
        let buffer_index = self.buffers.len();
        self.buffers.push(output_buffer);
        self.buffer_map.insert(node_index, buffer_index);

        node_index
    }

    pub fn connect(
        &mut self,
        from: GraphNode,
        to: GraphNode,
        channel_layout: ChannelLayout,
    ) -> Option<EdgeIndex> {
        match (from, to) {
            (GraphNode::Input, GraphNode::Node(node_idx)) => {
                self.input_edges.push((node_idx, channel_layout));
                None
            }
            (GraphNode::Input, GraphNode::Output) => {
                let passthrough_buffer = MultiChannelBuffer {
                    data: vec![0.0; 256].into_boxed_slice(), // TODO: buffer size should be stored in the graph
                };
                let passthrough_node = self.add_processor(PassThrough, passthrough_buffer);

                // Connect Input -> PassThrough -> Output
                self.input_edges
                    .push((passthrough_node, channel_layout.clone()));
                self.output_edges.push((passthrough_node, channel_layout));
                None
            }
            (GraphNode::Node(node_idx), GraphNode::Output) => {
                self.output_edges.push((node_idx, channel_layout));
                None
            }
            (GraphNode::Node(from_idx), GraphNode::Node(to_idx)) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(from_idx, to_idx, edge);
                self.recompute_topo_order();
                Some(edge_index)
            }
            // Invalid combinations
            (GraphNode::Input, GraphNode::Input)
            | (GraphNode::Node(_), GraphNode::Input)
            | (GraphNode::Output, _) => {
                panic!("Invalid connection: {:?} -> {:?}", from, to);
            }
        }
    }

    pub fn disconnect(&mut self, edge: EdgeIndex) {
        let _result = self.graph.remove_edge(edge);
        self.recompute_topo_order();
    }

    fn recompute_topo_order(&mut self) {
        // TODO: propagate graph error (e.g. cycles)
        self.topo_order = petgraph::algo::toposort(&self.graph, None)
            .expect("Graph has cycles! Ensure DAG structure.");
    }

    /// Processes the graph using provided input and output buffers.
    pub fn process(&mut self, input: &MultiChannelBuffer, output: &mut MultiChannelBuffer) {
        // Phase 1: Input phase - feed input to all entry nodes
        for (node_idx, channel_layout) in &self.input_edges {
            let output_buffer = self
                .buffer_map
                .get(node_idx)
                .and_then(|&i| self.buffers.get_mut(i))
                .unwrap();

            let processor_node = self.graph.node_weight_mut(*node_idx).unwrap();
            processor_node.process(input, output_buffer, channel_layout.clone());
        }

        // Phase 2: Process graph in topological order
        for &node_index in &self.topo_order {
            // Skip nodes that were already processed in input phase
            if self.input_edges.iter().any(|(idx, _)| *idx == node_index) {
                continue;
            }

            let num_incoming_edges = self
                .graph
                .edges_directed(node_index, Direction::Incoming)
                .count();

            // If there are multiple inputs, sum them
            if num_incoming_edges > 1 {
                self.summing_buffer.clear();
                for edge in self.graph.edges_directed(node_index, Direction::Incoming) {
                    let input_node = edge.source();
                    let input_buffer_index = self.buffer_map.get(&input_node).unwrap();
                    let input_buffer = &self.buffers[*input_buffer_index];
                    self.summing_buffer.add(input_buffer);
                }

                let output_buffer_index = self.buffer_map.get(&node_index).unwrap();
                let output_buffer = &mut self.buffers[*output_buffer_index];

                let processor_node = self.graph.node_weight_mut(node_index).unwrap();

                let mut context = ProcessingContext {
                    input_buffer: &self.summing_buffer,
                    output_buffer,
                    channel_layout: ChannelLayout {},
                };
                processor_node.processor.process(&mut context);
            } else if num_incoming_edges == 1 {
                let input_node = self
                    .graph
                    .neighbors_directed(node_index, Direction::Incoming)
                    .next()
                    .unwrap();

                let input_buffer_index = self.buffer_map.get(&input_node).unwrap();
                let output_buffer_index = self.buffer_map.get(&node_index).unwrap();

                let (low, high) = self.buffers.split_at_mut(*output_buffer_index);
                let input_buffer = &low[*input_buffer_index];
                let output_buffer = &mut high[0];

                let edge = self
                    .graph
                    .edges_directed(node_index, Direction::Incoming)
                    .next()
                    .unwrap();
                let channel_layout = edge.weight().get_layout();

                let processor_node = self.graph.node_weight_mut(node_index).unwrap();
                processor_node.processor.process(&mut ProcessingContext {
                    input_buffer,
                    output_buffer,
                    channel_layout,
                });
            }
            // If no incoming edges, this node was processed in input phase or is isolated
        }

        // Phase 3: Output phase - collect from all exit nodes
        output.clear();
        for (node_idx, _channel_layout) in &self.output_edges {
            let node_buffer = self
                .buffer_map
                .get(node_idx)
                .and_then(|&i| self.buffers.get(i))
                .unwrap();
            output.add(node_buffer);
        }
    }
}

struct FooProcessor {}
struct BarProcessor {}
struct BazProcessor {}

impl Processor for FooProcessor {
    fn process(&mut self, context: &mut ProcessingContext) {
        context.output_buffer.clear();
        println!("FooProcessor processing");
    }
}

impl Processor for BarProcessor {
    fn process(&mut self, _context: &mut ProcessingContext) {
        println!("BarProcessor processing");
    }
}

impl Processor for BazProcessor {
    fn process(&mut self, _context: &mut ProcessingContext) {
        println!("BazProcessor processing");
    }
}

fn main() {
    let mut graph = DspGraph::new();

    let foo = graph.add_processor(
        FooProcessor {},
        MultiChannelBuffer {
            data: vec![0.0; 256].into_boxed_slice(),
        },
    );
    let bar1 = graph.add_processor(
        BarProcessor {},
        MultiChannelBuffer {
            data: vec![0.0; 256].into_boxed_slice(),
        },
    );
    graph.connect(foo.into(), bar1.into(), ChannelLayout {});

    let baz = graph.add_processor(
        BazProcessor {},
        MultiChannelBuffer {
            data: vec![0.0; 256].into_boxed_slice(),
        },
    );
    graph.connect(bar1.into(), baz.into(), ChannelLayout {});

    let bar2 = graph.add_processor(
        BarProcessor {},
        MultiChannelBuffer {
            data: vec![0.0; 256].into_boxed_slice(),
        },
    );
    graph.connect(foo.into(), bar2.into(), ChannelLayout {});
    graph.connect(bar2.into(), baz.into(), ChannelLayout {});

    let bar3 = graph.add_processor(
        BarProcessor {},
        MultiChannelBuffer {
            data: vec![0.0; 256].into_boxed_slice(),
        },
    );
    graph.connect(foo.into(), bar3.into(), ChannelLayout {});
    graph.connect(bar3.into(), baz.into(), ChannelLayout {});

    // Connect input to foo and output from baz
    graph.connect(GraphNode::Input, foo.into(), ChannelLayout {});
    graph.connect(baz.into(), GraphNode::Output, ChannelLayout {});

    let input = MultiChannelBuffer {
        data: vec![0.0; 256].into_boxed_slice(),
    };
    let mut output = MultiChannelBuffer {
        data: vec![0.0; 256].into_boxed_slice(),
    };

    graph.process(&input, &mut output);
}
