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

    pub fn add(&mut self, other: &MultiChannelBuffer) {
        // Add other buffer to self
        println!("Adding buffer");
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a += *b);
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

pub struct NoOp;

impl Processor for NoOp {
    fn process(&mut self, _context: &mut ProcessingContext) {}
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
    input_node: NodeIndex,
    output_node: NodeIndex,
    summing_buffer: MultiChannelBuffer,
}

// TODO buffers and buffer_map could be merged into one HashMap<NodeIndex, MultiChannelBuffer>
impl DspGraph {
    pub fn new(buffer_size: usize) -> Self {
        let mut graph = DspGraph {
            graph: DiGraph::new(),
            topo_order: Vec::new(),
            buffers: Vec::new(),
            buffer_map: HashMap::new(),
            input_node: NodeIndex::end(),
            output_node: NodeIndex::end(),
            summing_buffer: MultiChannelBuffer::new(buffer_size),
        };

        graph.input_node = graph.add_processor(NoOp, MultiChannelBuffer::new(0));
        graph.output_node = graph.add_processor(NoOp, MultiChannelBuffer::new(0));

        graph
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

    // TODO: error handling, node indices must exist, return Result not Option
    pub fn connect(
        &mut self,
        from: GraphNode,
        to: GraphNode,
        channel_layout: ChannelLayout,
    ) -> Option<EdgeIndex> {
        let edge_index = match (from, to) {
            (GraphNode::Input, GraphNode::Node(node_idx)) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(self.input_node, node_idx, edge);
                Some(edge_index)
            }
            (GraphNode::Node(node_idx), GraphNode::Output) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(node_idx, self.output_node, edge);
                Some(edge_index)
            }
            (GraphNode::Node(from_idx), GraphNode::Node(to_idx)) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(from_idx, to_idx, edge);
                Some(edge_index)
            }
            (GraphNode::Input, GraphNode::Output) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(self.input_node, self.output_node, edge);
                let input_node = self.graph.node_weight_mut(self.input_node).unwrap();
                *input_node = ProcessorNode::new(PassThrough);
                Some(edge_index)
            }
            // Invalid combinations
            (GraphNode::Input, GraphNode::Input)
            | (GraphNode::Node(_), GraphNode::Input)
            | (GraphNode::Output, _) => {
                panic!("Invalid connection: {:?} -> {:?}", from, to);
            }
        };

        self.recompute_topo_order();
        edge_index
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

    pub fn process(&mut self, input: &MultiChannelBuffer, output: &mut MultiChannelBuffer) {
        output.clear();

        for &node_index in &self.topo_order {
            if node_index == self.output_node {
                continue;
            }

            let output_buffer_index = self.buffer_map.get(&node_index).unwrap();

            let num_incoming_edges = self
                .graph
                .edges_directed(node_index, Direction::Incoming)
                .count();

            // If there are multiple inputs, sum them
            if num_incoming_edges > 1 {
                self.summing_buffer.clear();
                for edge in self.graph.edges_directed(node_index, Direction::Incoming) {
                    let input_node = edge.source();
                    let input_buffer = if input_node == self.input_node {
                        input
                    } else {
                        let input_buffer_index = self.buffer_map.get(&input_node).unwrap();
                        &self.buffers[*input_buffer_index]
                    };
                    self.summing_buffer.add(input_buffer);
                }

                let output_buffer = &mut self.buffers[*output_buffer_index];

                let processor_node = self.graph.node_weight_mut(node_index).unwrap();

                // TODO: which layout?
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

                let (input_buffer, output_buffer) = if input_node == self.input_node {
                    let output_buffer = &mut self.buffers[*output_buffer_index];
                    (input, output_buffer)
                } else {
                    let input_buffer_index = self.buffer_map.get(&input_node).unwrap();
                    let (low, high) = self.buffers.split_at_mut(*output_buffer_index);
                    (&low[*input_buffer_index], &mut high[0])
                };

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
        }

        for node in self
            .graph
            .neighbors_directed(self.output_node, Direction::Incoming)
        {
            let input_buffer = if node == self.input_node {
                input
            } else {
                let input_buffer_index = self.buffer_map.get(&node).unwrap();
                &self.buffers[*input_buffer_index]
            };
            output.add(input_buffer);
        }
    }
}

struct FourtyTwo {}

impl Processor for FourtyTwo {
    fn process(&mut self, context: &mut ProcessingContext) {
        println!("FooProcessor processing");
        context.output_buffer.data.fill(42.0);
    }
}

#[test]
fn test_simple_graph() {
    let mut graph = DspGraph::new(10);
    let fourty_two = graph.add_processor(FourtyTwo {}, MultiChannelBuffer::new(10));
    graph.connect(GraphNode::Input, fourty_two.into(), ChannelLayout {});
    graph.connect(fourty_two.into(), GraphNode::Output, ChannelLayout {});

    let input = MultiChannelBuffer {
        data: vec![0.0; 10].into_boxed_slice(),
    };
    let mut output = MultiChannelBuffer {
        data: vec![0.0; 10].into_boxed_slice(),
    };
    graph.process(&input, &mut output);

    output.data.iter().for_each(|&x| {
        assert_eq!(x, 42.0);
    });
}

#[test]
fn test_passthrough() {
    let mut graph = DspGraph::new(10);
    graph.connect(GraphNode::Input, GraphNode::Output, ChannelLayout {});
    let input = MultiChannelBuffer {
        data: vec![2.0; 10].into_boxed_slice(),
    };
    let mut output = MultiChannelBuffer {
        data: vec![0.0; 10].into_boxed_slice(),
    };
    graph.process(&input, &mut output);

    output.data.iter().for_each(|&x| {
        assert_eq!(x, 2.0);
    });
}

#[test]
fn test_sum_at_output() {
    let mut graph = DspGraph::new(10);
    graph.connect(GraphNode::Input, GraphNode::Output, ChannelLayout {});
    let fourty_two = graph.add_processor(FourtyTwo {}, MultiChannelBuffer::new(10));
    graph.connect(GraphNode::Input, fourty_two.into(), ChannelLayout {});
    graph.connect(fourty_two.into(), GraphNode::Output, ChannelLayout {});

    let input = MultiChannelBuffer {
        data: vec![2.0; 10].into_boxed_slice(),
    };
    let mut output = MultiChannelBuffer {
        data: vec![0.0; 10].into_boxed_slice(),
    };

    graph.process(&input, &mut output);

    output.data.iter().for_each(|&x| {
        assert_eq!(x, 44.0);
    });
}

fn main() {}
