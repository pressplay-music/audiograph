//! Audiograph - a realtime audio processing graph library for Rust.
//!
//! Audiograph provides abstractions for audio processors, audio buffers and channel routing,
//! and enables the construction and management of directed signal processing graphs.
//!
//! # Building blocks
//!
//! [`DspGraph`] represents the main graph structure. It contains audio processing nodes and typed edges
//! that describe the signal flow between nodes. The major building blocks for constructing and using
//! a [`DspGraph`] are:
//!
//! - [`Processor`]: Trait representing an audio processing unit. Processors are the core of a graph node.
//! Conceptually, a graph node consists of a processor and its associated output buffer.
//! - [`AudioBuffer`]: Trait representing a buffer of audio samples organized by channels. There are multiple
//! implementations provided, including [`MultiChannelBuffer`] that owns channels of audio data and
//! [`MultiChannelBufferView`] as a non-owning alternative.
//! - [`ChannelLayout`]: Struct describing the active channels of a connection between nodes. The edges
//! of the graph carry optional channel layouts to indicate which channels of a node's output buffer
//! should be processed by connected successor nodes.
//! - [`ProcessingContext`]: Struct providing context for audio processing, including input and output buffer
//! references, channel layout and the number of frames to process. Used by a [`Processor`] as the source
//! of information for processing audio data.
//!
//! # Graph structure
//!
//! A [`DspGraph`] is a directed graph where nodes represent audio processors and edges represent
//! the signal flow between processors. It is possible for a node to have multiple incoming edges,
//! in which case the inputs are summed before being passed to the processor. Similarly, a node can have
//! multiple outgoing edges, allowing its output to be routed to multiple successor nodes.
//!
//! Edges of the graph are typed edges carrying additional information about the connection between nodes,
//! allowing complex routing scenarios such as channel selection and remapping.
//!
//! A graph can be designed and modified using the provided API, most notably using the following construction
//! methods:
//! - `add_processor`: Adds a new processor node to the graph along with its associated output buffer.
//! - `connect`: Connects two nodes in the graph with an edge, optionally specifying a channel layout.
//!
//! Additional methods are provided for more advanced operations, such as rewiring connections, enabling and
//! disabling edges and removing connections.
//!
//! Graph nodes and edges are identified using the `NodeIndex` and `EdgeIndex` types from the `petgraph` crate,
//! which is used as the underlying acyclic directed graph implementation.
//!
//! Input and output nodes: These are special nodes that serve as the entry and exit points of the graph.
//! Input and output nodes do not process any audio data and do not have dedicated buffers within the graph
//! structure. The corresponding buffers are passed to the [`DspGraph::process`] method instead. See also [`GraphNode`]
//! and [`DspGraph::connect`].
//!
//! # Realtime safety
//!
//! Realtime safety is guaranteed for most of the graph operations, including processing and modifying
//! the graph structure. Some operations, such as rewiring connections, are not realtime-safe and
//! indicated as such.
//!
//! Adding a node to the graph requires the node's audio buffer to be allocated. It is the caller's
//! responsibility to ensure this allocation is performed safely and not on the high-priority audio thread.
//!
//! Graph-internal memory allocations are performed upfront during graph initialization. Adding nodes
//! or edges within the specified graph capacity limits does not allocate.
//!

pub mod buffer;
pub mod channel;
pub mod processor;
pub mod sample;

pub use crate::buffer::*;
pub use crate::channel::*;
pub use crate::processor::*;
pub use crate::sample::*;

#[doc(no_inline)]
pub use petgraph::graph::{EdgeIndex, NodeIndex};

use petgraph::Direction;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{DfsPostOrder, EdgeRef, Reversed};

use std::collections::HashMap;

pub type AudioGraphError = &'static str;

/// The node type of the graph
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

#[derive(Clone)]
struct ProcessorChannel {
    pub channel_layout: Option<ChannelLayout>,
    pub has_rewire: bool,
    pub enabled: bool,
}

impl ProcessorChannel {
    pub fn new(channel_layout: Option<ChannelLayout>) -> Self {
        Self {
            channel_layout,
            has_rewire: false,
            enabled: true,
        }
    }

    pub fn get_layout(&self) -> Option<ChannelLayout> {
        self.channel_layout.clone()
    }
}

type GraphVisitMap<T> =
    <StableDiGraph<ProcessorNode<T>, ProcessorChannel> as petgraph::visit::Visitable>::Map;

pub struct DspGraph<T: Sample> {
    graph: StableDiGraph<ProcessorNode<T>, ProcessorChannel>,
    topo_order: Vec<NodeIndex>, // Pre-allocated processing order vector
    buffers: Vec<Option<MultiChannelBuffer<T>>>,
    input_node: NodeIndex,
    output_node: NodeIndex,
    summing_buffer: MultiChannelBuffer<T>,
    dfs_visitor: DfsPostOrder<NodeIndex, GraphVisitMap<T>>,
    topo_dirty: bool,
    edge_rewires: HashMap<EdgeIndex, HashMap<usize, usize>>,
}

impl<T: Sample> DspGraph<T> {
    pub fn new(num_channels: usize, frame_size: FrameSize, max_num_edges: Option<usize>) -> Self {
        let max_num_edges = max_num_edges.unwrap_or(64);

        let mut buffers = Vec::with_capacity(max_num_edges);
        for _ in 0..max_num_edges {
            buffers.push(None);
        }

        let mut graph = DspGraph {
            graph: StableDiGraph::with_capacity(max_num_edges, max_num_edges),
            topo_order: Vec::with_capacity(max_num_edges),
            buffers,
            input_node: NodeIndex::end(),
            output_node: NodeIndex::end(),
            summing_buffer: MultiChannelBuffer::new(num_channels, frame_size),
            dfs_visitor: DfsPostOrder::default(),
            topo_dirty: true,
            edge_rewires: HashMap::with_capacity(max_num_edges),
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
        if self.graph.node_count() >= self.graph.capacity().0 {
            return Err("Graph node capacity exceeded");
        }

        let node_index = self.graph.add_node(ProcessorNode::new(processor));
        let buffer_index = node_index.index();

        if buffer_index >= self.buffers.len() {
            return Err("Buffer capacity exceeded");
        }

        self.buffers[buffer_index] = Some(output_buffer);

        self.topo_dirty = true; // TODO: only if connected?

        Ok(node_index)
    }

    pub fn connect(
        &mut self,
        from: GraphNode,
        to: GraphNode,
        channel_layout: Option<ChannelLayout>,
    ) -> Result<EdgeIndex, AudioGraphError> {
        if self.graph.edge_count() >= self.graph.capacity().1 {
            return Err("Graph edge capacity exceeded");
        }

        // invalid combinations
        if let (GraphNode::Input, GraphNode::Input)
        | (GraphNode::Node(_), GraphNode::Input)
        | (GraphNode::Output, _) = (from, to)
        {
            return Err("Invalid connection");
        }

        let from_index = match from {
            GraphNode::Input => self.input_node,
            GraphNode::Output => self.output_node,
            GraphNode::Node(idx) => idx,
        };
        let to_index = match to {
            GraphNode::Input => self.input_node,
            GraphNode::Output => self.output_node,
            GraphNode::Node(idx) => idx,
        };

        if self.graph.node_weight(from_index).is_none() {
            return Err("Source node does not exist");
        }
        if self.graph.node_weight(to_index).is_none() {
            return Err("Destination node does not exist");
        }

        let channel_layout = match channel_layout {
            None => None,
            Some(layout) => {
                let mut channel_layout = layout;

                if from_index != self.input_node {
                    let source_buffer_channels = self.buffers[from_index.index()]
                        .as_ref()
                        .unwrap()
                        .num_channels();
                    channel_layout.clamp(source_buffer_channels);
                }

                if to_index != self.output_node {
                    let destination_buffer_channels = self.buffers[to_index.index()]
                        .as_ref()
                        .unwrap()
                        .num_channels();
                    channel_layout.clamp(destination_buffer_channels);
                }
                Some(channel_layout)
            }
        };

        let edge = ProcessorChannel::new(channel_layout);

        let edge_index = match self.graph.find_edge(from_index, to_index) {
            Some(existing_edge_index) => {
                *self.graph.edge_weight_mut(existing_edge_index).unwrap() = edge;
                existing_edge_index
            }
            None => self.graph.add_edge(from_index, to_index, edge),
        };

        self.topo_dirty = true;
        Ok(edge_index)
    }

    /// NOT realtime-safe
    pub fn rewire(
        &mut self,
        edge_index: EdgeIndex,
        rewire_mapping: &[(usize, usize)], // maps source channel to destination channel
    ) -> Result<(), AudioGraphError> {
        if let Some(edge) = self.graph.edge_weight_mut(edge_index) {
            edge.has_rewire = true;

            let mut channel_layout = ChannelLayout::new(0);
            let rewire = self.edge_rewires.entry(edge_index).or_default();
            rewire.clear();

            for &(source, dest) in rewire_mapping {
                channel_layout.connect(dest);

                // we flip (source, dest) to have logical to physical mapping
                rewire.insert(dest, source);
            }

            edge.channel_layout = Some(channel_layout);

            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    /// NOT realtime-safe
    pub fn remove_rewire(&mut self, edge_index: EdgeIndex) -> Result<(), AudioGraphError> {
        if let Some(edge) = self.graph.edge_weight_mut(edge_index) {
            edge.has_rewire = false;
            self.edge_rewires.remove(&edge_index);
            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    /// NOT realtime-safe
    pub fn connect_rewired(
        &mut self,
        from: GraphNode,
        to: GraphNode,
        wiring: &[(usize, usize)],
    ) -> Result<EdgeIndex, AudioGraphError> {
        let edge = self.connect(from, to, None)?;
        self.rewire(edge, wiring)?;

        Ok(edge)
    }

    pub fn remove_connection(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        // Note: Rewire information (if any) is deliberately leaked for RT-safety
        self.graph.remove_edge(edge).ok_or("Edge not found")?;
        self.topo_dirty = true;
        Ok(())
    }

    pub fn enable_connection(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        if let Some(edge_weight) = self.graph.edge_weight_mut(edge) {
            edge_weight.enabled = true;
            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    pub fn disable_connection(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        if let Some(edge_weight) = self.graph.edge_weight_mut(edge) {
            edge_weight.enabled = false;
            Ok(())
        } else {
            Err("Edge not found")
        }
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

    pub fn process(
        &mut self,
        input: &dyn AudioBuffer<T>,
        output: &mut dyn AudioBuffer<T>,
        num_frames: FrameSize,
    ) {
        self.ensure_topo_order_updated(); // update node order if necessary

        output.clear();

        for &node_index in &self.topo_order {
            if node_index == self.output_node {
                continue;
            }

            let output_buffer_index = node_index.index();

            let mut incoming_edges = self
                .graph
                .edges_directed(node_index, Direction::Incoming)
                .filter(|edge| edge.weight().enabled);

            let num_incoming_edges = incoming_edges.clone().count();

            // If there are multiple inputs, sum them
            if num_incoming_edges > 1 {
                self.summing_buffer.clear();
                let mut channel_layout: Option<ChannelLayout> = None;

                for edge in incoming_edges {
                    let input_node = edge.source();
                    let input_buffer: &dyn AudioBuffer<T> = if input_node == self.input_node {
                        input
                    } else {
                        let input_buffer_index = input_node.index();
                        self.buffers[input_buffer_index].as_ref().unwrap()
                    };

                    let edge_layout = edge.weight().get_layout();

                    if edge.weight().has_rewire {
                        if let Some(rewire) = self.edge_rewires.get(&edge.id()) {
                            let rewired_buffer_view = RewiredBufferView {
                                buffer: input_buffer,
                                rewire,
                            };
                            self.summing_buffer.add(&rewired_buffer_view, &edge_layout);
                        } else {
                            self.summing_buffer.add(input_buffer, &edge_layout);
                        }
                    } else {
                        self.summing_buffer.add(input_buffer, &edge_layout);
                    }

                    if let Some(edge_layout) = &edge_layout {
                        if let Some(existing_layout) = &mut channel_layout {
                            existing_layout.combine(edge_layout);
                        } else {
                            channel_layout = Some(edge_layout.clone());
                        }
                    }
                }

                let output_buffer: &mut dyn AudioBuffer<T> =
                    self.buffers[output_buffer_index].as_mut().unwrap();
                let processor_node = self.graph.node_weight_mut(node_index).unwrap();

                let mut context = ProcessingContext::create_unchecked(
                    &self.summing_buffer,
                    output_buffer,
                    channel_layout,
                    num_frames,
                );

                processor_node.process(&mut context);
            } else if num_incoming_edges == 1 {
                let (input_node, channel_layout, has_rewire, edge_id) = {
                    let edge = incoming_edges.next().unwrap();
                    (
                        edge.source(),
                        edge.weight().get_layout(),
                        edge.weight().has_rewire,
                        edge.id(),
                    )
                };

                let (input_buffer, output_buffer): (&dyn AudioBuffer<T>, &mut dyn AudioBuffer<T>) =
                    if input_node == self.input_node {
                        let output_buffer = self.buffers[output_buffer_index].as_mut().unwrap();
                        (input, output_buffer)
                    } else {
                        let input_buffer_index = input_node.index();
                        let (low, high) = self.buffers.split_at_mut(output_buffer_index);
                        (
                            low[input_buffer_index].as_ref().unwrap(),
                            high[0].as_mut().unwrap(),
                        )
                    };

                let processor_node = self.graph.node_weight_mut(node_index).unwrap();

                if has_rewire {
                    let rewired_buffer_view = RewiredBufferView {
                        buffer: input_buffer,
                        rewire: self.edge_rewires.get(&edge_id).unwrap(),
                    };
                    let mut context = ProcessingContext::create_unchecked(
                        &rewired_buffer_view,
                        output_buffer,
                        channel_layout,
                        num_frames,
                    );
                    processor_node.process(&mut context);
                } else {
                    let mut context = ProcessingContext::create_unchecked(
                        input_buffer,
                        output_buffer,
                        channel_layout,
                        num_frames,
                    );
                    processor_node.process(&mut context);
                }
            }
        }

        for edge in self
            .graph
            .edges_directed(self.output_node, Direction::Incoming)
            .filter(|e| e.weight().enabled)
        {
            let node = edge.source();
            let node_buffer: &dyn AudioBuffer<T> = if node == self.input_node {
                input
            } else {
                let input_buffer_index = node.index();
                self.buffers[input_buffer_index].as_ref().unwrap()
            };
            // TODO: handle disconnected channels
            output.add(node_buffer, &edge.weight().get_layout());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::processor::PassThrough;

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
        graph.process(&input, &mut output, FrameSize(10));

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
        graph.process(&input, &mut output, FrameSize(10));

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

        graph.process(&input, &mut output, FrameSize(10));

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
        graph.process(&input, &mut output, FrameSize(10));

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

        graph.process(&input, &mut output, FrameSize(8));

        output.channel(0).unwrap().iter().for_each(|&x| {
            assert_eq!(x, 10.0);
        });
    }

    #[test]
    fn test_reconnect() {
        let frame_size = FrameSize(10);

        let mut graph = DspGraph::<f32>::new(2, frame_size, Some(16));

        let node_a = graph
            .add_processor(PassThrough {}, MultiChannelBuffer::new(2, frame_size))
            .unwrap();

        // Initial connection
        let input_edge = graph
            .connect(GraphNode::Input, node_a.into(), Some(ChannelLayout::new(2)))
            .unwrap();

        let output_edge = graph
            .connect(
                node_a.into(),
                GraphNode::Output,
                Some(ChannelLayout::new(2)),
            )
            .unwrap();

        let mut input = MultiChannelBuffer::new(2, frame_size);
        input.channel_mut(0).unwrap().fill(1.0);
        input.channel_mut(1).unwrap().fill(2.0);

        let mut output = MultiChannelBuffer::new(2, frame_size);

        graph.process(&input, &mut output, frame_size);
        assert_eq!(output.channel(0).unwrap()[0], 1.0);
        assert_eq!(output.channel(1).unwrap()[0], 2.0);

        // reconnect with different layout
        let new_input_edge = graph
            .connect(
                GraphNode::Input,
                node_a.into(),
                Some(ChannelLayout::from_indices(&[0])), // Only channel 0
            )
            .unwrap();
        let new_output_edge = graph
            .connect(
                node_a.into(),
                GraphNode::Output,
                Some(ChannelLayout::from_indices(&[0])), // Only channel 0
            )
            .unwrap();

        assert_eq!(input_edge, new_input_edge);
        assert_eq!(output_edge, new_output_edge);

        output.clear();
        graph.process(&input, &mut output, frame_size);
        assert_eq!(output.channel(0).unwrap()[0], 1.0);
        assert_eq!(output.channel(1).unwrap()[0], 0.0); // Channel 1 disconnected

        graph
            .connect(
                GraphNode::Input,
                node_a.into(),
                Some(ChannelLayout::from_indices(&[1])), // Only channel 1
            )
            .unwrap();
        graph
            .connect(
                node_a.into(),
                GraphNode::Output,
                Some(ChannelLayout::from_indices(&[1])), // Only channel 1
            )
            .unwrap();

        output.clear();
        graph.process(&input, &mut output, frame_size);
        assert_eq!(output.channel(0).unwrap()[0], 0.0); // Channel 0 disconnected
        assert_eq!(output.channel(1).unwrap()[0], 2.0);
    }

    #[test]
    fn test_capacity_limits() {
        let mut graph = DspGraph::<f32>::new(1, FrameSize(10), Some(4));

        // Capacity is 4, input and output are already created (2 nodes), so we can add 2 more
        let n1 = graph
            .add_processor(PassThrough {}, MultiChannelBuffer::new(1, FrameSize(10)))
            .unwrap();
        let n2 = graph
            .add_processor(PassThrough {}, MultiChannelBuffer::new(1, FrameSize(10)))
            .unwrap();

        assert!(
            graph
                .add_processor(PassThrough {}, MultiChannelBuffer::new(1, FrameSize(10)))
                .is_err()
        );

        graph.connect(GraphNode::Input, n1.into(), None).unwrap();
        graph.connect(n1.into(), n2.into(), None).unwrap();
        graph.connect(n2.into(), GraphNode::Output, None).unwrap();
        graph.connect(GraphNode::Input, n2.into(), None).unwrap();

        let result = graph.connect(n1.into(), GraphNode::Output, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_enable_disable_connection() {
        let frame_size = FrameSize(10);
        let mut graph = DspGraph::<f32>::new(1, frame_size, None);

        let node = graph
            .add_processor(FourtyTwo {}, MultiChannelBuffer::new(1, frame_size))
            .unwrap();

        graph.connect(GraphNode::Input, node.into(), None).unwrap();

        let output_edge = graph.connect(node.into(), GraphNode::Output, None).unwrap();

        let mut input = MultiChannelBuffer::new(1, frame_size);
        input.channel_mut(0).unwrap().fill(1.0);

        let mut output = MultiChannelBuffer::new(1, frame_size);

        graph.process(&input, &mut output, frame_size);
        assert_eq!(output.channel(0).unwrap()[0], 42.0);

        graph.disable_connection(output_edge).unwrap();

        output.clear();
        graph.process(&input, &mut output, frame_size);
        assert_eq!(output.channel(0).unwrap()[0], 0.0);

        graph.enable_connection(output_edge).unwrap();

        output.clear();
        graph.process(&input, &mut output, frame_size);
        assert_eq!(output.channel(0).unwrap()[0], 42.0);
    }
}
