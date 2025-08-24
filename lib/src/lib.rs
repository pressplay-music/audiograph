mod buffer;
mod channel;
mod processor;
mod sample;

use crate::buffer::{AudioBuffer, FrameSize, MultiChannelBuffer};
use crate::channel::ChannelLayout;
use crate::processor::{NoOp, PassThrough, ProcessingContext, Processor};
use crate::sample::Sample;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{DfsPostOrder, EdgeRef, Reversed};
use petgraph::Direction;

use std::collections::HashMap;

pub type AudioGraphError = &'static str;

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

type GraphVisitMap<T> =
    <StableDiGraph<ProcessorNode<T>, ProcessorChannel> as petgraph::visit::Visitable>::Map;

pub struct DspGraph<T: Sample> {
    graph: StableDiGraph<ProcessorNode<T>, ProcessorChannel>,
    topo_order: Vec<NodeIndex>, // Pre-allocated processing order vector
    buffers: Vec<MultiChannelBuffer<T>>,
    buffer_map: HashMap<NodeIndex, usize>,
    input_node: NodeIndex,
    output_node: NodeIndex,
    summing_buffer: MultiChannelBuffer<T>,
    dfs_visitor: DfsPostOrder<NodeIndex, GraphVisitMap<T>>,
    topo_dirty: bool,
}

impl<T: Sample> DspGraph<T> {
    pub fn new(num_channels: usize, frame_size: FrameSize, max_num_edges: Option<usize>) -> Self {
        let max_num_edges = max_num_edges.unwrap_or(64);
        let mut graph = DspGraph {
            graph: StableDiGraph::with_capacity(max_num_edges, max_num_edges),
            topo_order: Vec::with_capacity(max_num_edges),
            buffers: Vec::with_capacity(max_num_edges),
            buffer_map: HashMap::with_capacity(max_num_edges),
            input_node: NodeIndex::end(),
            output_node: NodeIndex::end(),
            summing_buffer: MultiChannelBuffer::new(num_channels, frame_size),
            dfs_visitor: DfsPostOrder::default(),
            topo_dirty: true,
        };

        graph.input_node = graph
            .add_processor(NoOp, MultiChannelBuffer::new(0, FrameSize(0)))
            .unwrap();
        graph.output_node = graph
            .add_processor(NoOp, MultiChannelBuffer::new(0, FrameSize(0)))
            .unwrap();

        graph.dfs_visitor = DfsPostOrder::empty(&graph.graph);

        graph
    }

    pub fn add_processor<P: Processor<T> + Send + 'static>(
        &mut self,
        processor: P,
        output_buffer: MultiChannelBuffer<T>,
    ) -> Result<NodeIndex, AudioGraphError> {
        let buffer_index = self.buffers.len();

        if buffer_index >= self.buffers.capacity() {
            return Err("Buffer capacity exceeded");
        }

        let node_index = self.graph.add_node(ProcessorNode::new(processor));

        self.buffers.push(output_buffer);
        self.buffer_map.insert(node_index, buffer_index);

        self.topo_dirty = true;

        Ok(node_index)
    }

    pub fn connect(
        &mut self,
        from: GraphNode,
        to: GraphNode,
        channel_layout: Option<ChannelLayout>,
    ) -> Result<EdgeIndex, AudioGraphError> {
        let channel_layout = channel_layout.unwrap_or_default();
        let edge_index = match (from, to) {
            (GraphNode::Input, GraphNode::Node(node_idx)) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(self.input_node, node_idx, edge);
                Ok(edge_index)
            }
            (GraphNode::Node(node_idx), GraphNode::Output) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(node_idx, self.output_node, edge);
                Ok(edge_index)
            }
            (GraphNode::Node(from_idx), GraphNode::Node(to_idx)) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(from_idx, to_idx, edge);
                Ok(edge_index)
            }
            (GraphNode::Input, GraphNode::Output) => {
                let edge = ProcessorChannel { channel_layout };
                let edge_index = self.graph.add_edge(self.input_node, self.output_node, edge);
                let input_node = self.graph.node_weight_mut(self.input_node).unwrap();
                *input_node = ProcessorNode::new(PassThrough);
                Ok(edge_index)
            }
            // Invalid combinations
            (GraphNode::Input, GraphNode::Input)
            | (GraphNode::Node(_), GraphNode::Input)
            | (GraphNode::Output, _) => Err("Invalid connection"),
        };

        self.topo_dirty = true;
        edge_index
    }

    pub fn disconnect(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        self.graph.remove_edge(edge).ok_or("Edge not found")?;
        self.topo_dirty = true;
        Ok(())
    }

    fn ensure_topo_order_updated(&mut self) {
        if self.topo_dirty {
            self.topo_order.clear(); // clear but keep preallocated size

            self.dfs_visitor.reset(&self.graph);
            let reversed = Reversed(&self.graph);
            for start_node in self
                .graph
                .neighbors_directed(self.output_node, Direction::Incoming)
            {
                self.dfs_visitor.move_to(start_node);
                while let Some(node) = self.dfs_visitor.next(&reversed) {
                    self.topo_order.push(node);
                }
            }

            self.topo_dirty = false;
        }
    }

    pub fn process(&mut self, input: &dyn AudioBuffer<T>, output: &mut dyn AudioBuffer<T>) {
        self.ensure_topo_order_updated(); // update node order if necessary

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
mod tests {
    use super::*;

    struct FourtyTwo {}

    impl Processor<f32> for FourtyTwo {
        fn process(&mut self, context: &mut ProcessingContext<f32>) {
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
        let mut graph = DspGraph::<f32>::new(1, FrameSize(10), None);
        let fourty_two = graph
            .add_processor(FourtyTwo {}, MultiChannelBuffer::new(1, FrameSize(10)))
            .unwrap();
        graph
            .connect(GraphNode::Input, fourty_two.into(), None)
            .unwrap();
        graph
            .connect(fourty_two.into(), GraphNode::Output, None)
            .unwrap();

        let input = MultiChannelBuffer::new(1, FrameSize(10));
        let mut output = MultiChannelBuffer::new(1, FrameSize(10));
        graph.process(&input, &mut output);

        output.channel(0).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 42.0);
        });
    }

    #[test]
    fn test_passthrough() {
        let mut graph = DspGraph::<f32>::new(1, FrameSize(10), None);
        graph
            .connect(GraphNode::Input, GraphNode::Output, None)
            .unwrap();
        let mut input = MultiChannelBuffer::new(1, FrameSize(10));
        input.channel_mut(0).unwrap().fill(2.0);

        let mut output = MultiChannelBuffer::new(1, FrameSize(10));
        graph.process(&input, &mut output);

        output.channel(0).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 2.0);
        });
    }

    #[test]
    fn test_sum_at_output() {
        let mut graph = DspGraph::<f32>::new(1, FrameSize(10), None);
        graph
            .connect(GraphNode::Input, GraphNode::Output, None)
            .unwrap();
        let fourty_two = graph
            .add_processor(FourtyTwo {}, MultiChannelBuffer::new(1, FrameSize(10)))
            .unwrap();
        graph
            .connect(GraphNode::Input, fourty_two.into(), None)
            .unwrap();
        graph
            .connect(fourty_two.into(), GraphNode::Output, None)
            .unwrap();

        let mut input = MultiChannelBuffer::new(1, FrameSize(10));
        input.channel_mut(0).unwrap().fill(2.0);
        let mut output = MultiChannelBuffer::new(1, FrameSize(10));

        graph.process(&input, &mut output);

        output.channel(0).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 44.0);
        });
    }

    #[test]
    fn test_partial_channel_routing() {
        let mut graph = DspGraph::<f32>::new(3, FrameSize(10), None);
        let fourty_two = graph
            .add_processor(FourtyTwo {}, MultiChannelBuffer::new(3, FrameSize(10)))
            .unwrap();

        let passthrough = graph
            .add_processor(PassThrough {}, MultiChannelBuffer::new(3, FrameSize(10)))
            .unwrap();

        // all 3 channels connected
        graph
            .connect(GraphNode::Input, fourty_two.into(), None)
            .unwrap();

        let mut second_channel_only = ChannelLayout::new(0); // Start with no channels
        second_channel_only.connect(1);

        graph
            .connect(
                fourty_two.into(),
                passthrough.into(),
                Some(second_channel_only),
            )
            .unwrap();

        graph
            .connect(passthrough.into(), GraphNode::Output, None)
            .unwrap();

        let mut input = MultiChannelBuffer::new(3, FrameSize(10));
        input.channel_mut(0).unwrap().fill(1.0);
        input.channel_mut(1).unwrap().fill(2.0);
        input.channel_mut(2).unwrap().fill(3.0);

        let mut output = MultiChannelBuffer::new(3, FrameSize(10));
        graph.process(&input, &mut output);

        // Only channel 1 should have the FourtyTwo output (42.0)
        output.channel(0).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 0.0);
        });
        output.channel(1).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 42.0);
        });
        output.channel(2).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 0.0);
        });
    }

    #[test]
    fn test_processing_order_and_count() {
        struct Adder {
            value: f32,
        }

        impl Processor<f32> for Adder {
            fn process(&mut self, context: &mut ProcessingContext<f32>) {
                for ch in 0..context.output_buffer.num_channels() {
                    let input_channel = context.input_buffer.channel(ch).unwrap();
                    let output_channel = context.output_buffer.channel_mut(ch).unwrap();
                    for (o, &i) in output_channel.iter_mut().zip(input_channel.iter()) {
                        *o = i + self.value;
                    }
                }
            }
        }

        let mut graph = DspGraph::<f32>::new(1, FrameSize(8), None);

        let add1 = graph
            .add_processor(
                Adder { value: 1.0 },
                MultiChannelBuffer::new(1, FrameSize(8)),
            )
            .unwrap();

        let add3 = graph
            .add_processor(
                Adder { value: 3.0 },
                MultiChannelBuffer::new(1, FrameSize(8)),
            )
            .unwrap();
        let add5 = graph
            .add_processor(
                Adder { value: 5.0 },
                MultiChannelBuffer::new(1, FrameSize(8)),
            )
            .unwrap();

        graph.connect(GraphNode::Input, add1.into(), None).unwrap();
        graph.connect(add1.into(), add3.into(), None).unwrap();
        graph.connect(add1.into(), add5.into(), None).unwrap();
        graph.connect(add3.into(), GraphNode::Output, None).unwrap();
        graph.connect(add5.into(), GraphNode::Output, None).unwrap();

        let input = MultiChannelBuffer::new(1, FrameSize(8));
        let mut output = MultiChannelBuffer::new(1, FrameSize(8));

        graph.process(&input, &mut output);

        output.channel(0).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 10.0);
        });
    }
}
