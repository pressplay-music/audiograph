mod buffer;
mod channel;
mod processor;
mod sample;

use crate::buffer::{AudioBuffer, MultiChannelBuffer};
use crate::channel::ChannelLayout;
use crate::processor::{NoOp, PassThrough, ProcessingContext, Processor};
use crate::sample::Sample;

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

/// Represents an audio processing node
struct ProcessorNode<T: Sample> {
    processor: Box<dyn Processor<T> + Send>,
}

impl<T: Sample> ProcessorNode<T> {
    pub fn new(processor: impl Processor<T> + Send + 'static) -> Self {
        Self {
            processor: Box::new(processor),
        }
    }

    pub fn process(&mut self, context: &mut ProcessingContext<T>) {
        self.processor.process(context);
    }
}

struct ProcessorChannel {
    pub channel_layout: ChannelLayout,
}

impl ProcessorChannel {
    pub fn get_layout(&self) -> ChannelLayout {
        self.channel_layout.clone()
    }
}

pub struct DspGraph<T: Sample> {
    graph: DiGraph<ProcessorNode<T>, ProcessorChannel>,
    topo_order: Vec<NodeIndex>, // Precomputed processing order
    buffers: Vec<MultiChannelBuffer<T>>,
    buffer_map: HashMap<NodeIndex, usize>,
    input_node: NodeIndex,
    output_node: NodeIndex,
    summing_buffer: MultiChannelBuffer<T>,
}

impl<T: Sample> DspGraph<T> {
    // TODO: change argument order
    pub fn new(frame_size: usize, num_channels: usize) -> Self {
        let mut graph = DspGraph {
            graph: DiGraph::new(),
            topo_order: Vec::new(),
            buffers: Vec::new(),
            buffer_map: HashMap::new(),
            input_node: NodeIndex::end(),
            output_node: NodeIndex::end(),
            summing_buffer: MultiChannelBuffer::new(num_channels, frame_size),
        };

        graph.input_node = graph.add_processor(NoOp, MultiChannelBuffer::new(0, 0));
        graph.output_node = graph.add_processor(NoOp, MultiChannelBuffer::new(0, 0));

        graph
    }

    pub fn add_processor<P: Processor<T> + Send + 'static>(
        &mut self,
        processor: P,
        output_buffer: MultiChannelBuffer<T>,
    ) -> NodeIndex {
        let node_index = self.graph.add_node(ProcessorNode::new(processor));

        // Store buffer and map NodeIndex -> Vec index
        let buffer_index = self.buffers.len();
        self.buffers.push(output_buffer);
        self.buffer_map.insert(node_index, buffer_index);

        node_index
    }

    // TODO: error handling, node indices must exist, return Result not Option
    // TODO: channel layout optional?
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

    pub fn process(&mut self, input: &dyn AudioBuffer<T>, output: &mut dyn AudioBuffer<T>) {
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
                    let input_buffer: &dyn AudioBuffer<T> = if input_node == self.input_node {
                        input
                    } else {
                        let input_buffer_index = self.buffer_map.get(&input_node).unwrap();
                        &self.buffers[*input_buffer_index]
                    };

                    // TODO: handle disconnected channels?
                    self.summing_buffer
                        .add(input_buffer, edge.weight().get_layout());
                }

                let output_buffer: &mut dyn AudioBuffer<T> =
                    &mut self.buffers[*output_buffer_index];
                let channel_layout = self
                    .graph
                    .edges_directed(node_index, Direction::Incoming)
                    .next()
                    .unwrap()
                    .weight()
                    .get_layout();
                let processor_node = self.graph.node_weight_mut(node_index).unwrap();

                let mut context = ProcessingContext {
                    input_buffer: &self.summing_buffer,
                    output_buffer,
                    channel_layout,
                };
                processor_node.process(&mut context);
            } else if num_incoming_edges == 1 {
                let input_node = self
                    .graph
                    .neighbors_directed(node_index, Direction::Incoming)
                    .next()
                    .unwrap();

                let (input_buffer, output_buffer): (&dyn AudioBuffer<T>, &mut dyn AudioBuffer<T>) =
                    if input_node == self.input_node {
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
                let mut context = ProcessingContext {
                    input_buffer,
                    output_buffer,
                    channel_layout,
                };
                processor_node.process(&mut context);
            }
        }

        for edge in self
            .graph
            .edges_directed(self.output_node, Direction::Incoming)
        {
            let node = edge.source();
            let node_buffer: &dyn AudioBuffer<T> = if node == self.input_node {
                input
            } else {
                let input_buffer_index = self.buffer_map.get(&node).unwrap();
                &self.buffers[*input_buffer_index]
            };
            // TODO: handle disconnected channels
            output.add(node_buffer, edge.weight().get_layout());
        }
    }
}

#[cfg(test)]
struct FourtyTwo {}

#[cfg(test)]
impl Processor<f32> for FourtyTwo {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        println!("FooProcessor processing");
        for channel in 0..context.output_buffer.num_channels() {
            context
                .output_buffer
                .channel_mut(channel)
                .unwrap()
                .fill(42.0);
        }
    }
}

#[test]
fn test_simple_graph() {
    let mut graph = DspGraph::<f32>::new(10, 1);
    let fourty_two = graph.add_processor(FourtyTwo {}, MultiChannelBuffer::new(1, 10));
    graph.connect(
        GraphNode::Input,
        fourty_two.into(),
        ChannelLayout::default(),
    );
    graph.connect(
        fourty_two.into(),
        GraphNode::Output,
        ChannelLayout::default(),
    );

    let input = MultiChannelBuffer::new(1, 10);
    let mut output = MultiChannelBuffer::new(1, 10);
    graph.process(&input, &mut output);

    output.channel(0).unwrap().iter().for_each(|&x| {
        assert_eq!(x, 42.0);
    });
}

#[test]
fn test_passthrough() {
    let mut graph = DspGraph::<f32>::new(10, 1);
    graph.connect(
        GraphNode::Input,
        GraphNode::Output,
        ChannelLayout::default(),
    );
    let mut input = MultiChannelBuffer::new(1, 10);
    input.channel_mut(0).unwrap().fill(2.0);

    let mut output = MultiChannelBuffer::new(1, 10);
    graph.process(&input, &mut output);

    output.channel(0).unwrap().iter().for_each(|&x| {
        assert_eq!(x, 2.0);
    });
}

#[test]
fn test_sum_at_output() {
    let mut graph = DspGraph::<f32>::new(10, 1);
    graph.connect(
        GraphNode::Input,
        GraphNode::Output,
        ChannelLayout::default(),
    );
    let fourty_two = graph.add_processor(FourtyTwo {}, MultiChannelBuffer::new(1, 10));
    graph.connect(
        GraphNode::Input,
        fourty_two.into(),
        ChannelLayout::default(),
    );
    graph.connect(
        fourty_two.into(),
        GraphNode::Output,
        ChannelLayout::default(),
    );

    let mut input = MultiChannelBuffer::new(1, 10);
    input.channel_mut(0).unwrap().fill(2.0);
    let mut output = MultiChannelBuffer::new(1, 10);

    graph.process(&input, &mut output);

    output.channel(0).unwrap().iter().for_each(|&x| {
        assert_eq!(x, 44.0);
    });
}
