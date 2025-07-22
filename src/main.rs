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

pub trait Sample: num::Float + Default + std::ops::Add + std::ops::AddAssign {}
impl<T> Sample for T where T: num::Float + Default + std::ops::Add + std::ops::AddAssign {}

// TODO: add channel iterators and more creation methods
// TODO: clear() return type
pub trait AudioBuffer<T: Sample> {
    fn num_channels(&self) -> usize;
    fn num_frames(&self) -> usize;
    fn channel(&self, index: usize) -> Option<&[T]>;
    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]>;
    fn clear(&mut self);
    fn add(&mut self, other: &dyn AudioBuffer<T>, _channel_layout: ChannelLayout) {
        for channel in 0..self.num_channels() {
            if let (Some(self_channel), Some(other_channel)) =
                (self.channel_mut(channel), other.channel(channel))
            {
                self_channel
                    .iter_mut()
                    .zip(other_channel.iter())
                    .for_each(|(a, b)| {
                        *a += *b;
                    });
            }
        }
    }
}

pub struct MultiChannelBuffer<T> {
    channels: Vec<Box<[T]>>,
    num_frames: usize,
}

impl<T: Sample> MultiChannelBuffer<T> {
    pub fn new(num_channels: usize, num_frames: usize) -> Self {
        let mut channels = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            channels.push(vec![T::zero(); num_frames].into_boxed_slice());
        }
        Self {
            channels,
            num_frames,
        }
    }
}

impl<T: Sample> AudioBuffer<T> for MultiChannelBuffer<T> {
    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn channel(&self, index: usize) -> Option<&[T]> {
        self.channels.get(index).map(|b| &**b)
    }

    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.channels.get_mut(index).map(|b| &mut **b)
    }

    fn clear(&mut self) {
        for channel in self.channels.iter_mut() {
            for sample in channel.iter_mut() {
                *sample = T::zero();
            }
        }
    }
}

pub struct MultiChannelBufferView<'a, T: Sample> {
    channels: &'a [Box<[T]>],
    num_frames: usize,
}

impl<T: Sample> AudioBuffer<T> for MultiChannelBufferView<'_, T> {
    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn channel(&self, index: usize) -> Option<&[T]> {
        self.channels.get(index).map(|b| &**b)
    }

    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        None
    }

    fn clear(&mut self) {}
}

pub struct ProcessingContext<'a, T: Sample> {
    pub input_buffer: &'a dyn AudioBuffer<T>,
    pub output_buffer: &'a mut dyn AudioBuffer<T>,
    pub channel_layout: ChannelLayout,
}

#[derive(Clone)]
pub struct ChannelLayout {}

impl ChannelLayout {
    pub fn compatible(&self, _other: &ChannelLayout) -> bool {
        // TODO: implement meaningful check
        true
    }
}

pub trait Processor<T: Sample> {
    fn process(&mut self, context: &mut ProcessingContext<T>);
}

pub struct PassThrough;

impl<T: Sample> Processor<T> for PassThrough {
    // TODO: find way to implement channel iterators
    fn process(&mut self, context: &mut ProcessingContext<T>) {
        for channel in 0..context.input_buffer.num_channels() {
            if let Some(input_channel) = context.input_buffer.channel(channel) {
                if let Some(output_channel) = context.output_buffer.channel_mut(channel) {
                    output_channel.copy_from_slice(input_channel);
                }
            }
        }
    }
}

pub struct NoOp;

impl<T: Sample> Processor<T> for NoOp {
    fn process(&mut self, _context: &mut ProcessingContext<T>) {}
}

/// Represents an audio processing node
pub struct ProcessorNode<T: Sample> {
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

pub struct ProcessorChannel {
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
            output.add(node_buffer, edge.weight().get_layout());
        }
    }
}

pub struct FourtyTwo {}

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
    graph.connect(GraphNode::Input, fourty_two.into(), ChannelLayout {});
    graph.connect(fourty_two.into(), GraphNode::Output, ChannelLayout {});

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
    graph.connect(GraphNode::Input, GraphNode::Output, ChannelLayout {});
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
    graph.connect(GraphNode::Input, GraphNode::Output, ChannelLayout {});
    let fourty_two = graph.add_processor(FourtyTwo {}, MultiChannelBuffer::new(1, 10));
    graph.connect(GraphNode::Input, fourty_two.into(), ChannelLayout {});
    graph.connect(fourty_two.into(), GraphNode::Output, ChannelLayout {});

    let mut input = MultiChannelBuffer::new(1, 10);
    input.channel_mut(0).unwrap().fill(2.0);
    let mut output = MultiChannelBuffer::new(1, 10);

    graph.process(&input, &mut output);

    output.channel(0).unwrap().iter().for_each(|&x| {
        assert_eq!(x, 44.0);
    });
}

fn main() {}
