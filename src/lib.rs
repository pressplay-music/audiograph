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
//!   Conceptually, a graph node consists of a processor and its associated output buffer.
//! - [`AudioBuffer`]: Trait representing a buffer of audio samples organized by channels. Multiple
//!   implementations are provided, including [`MultiChannelBuffer`] (owning) and
//!   [`MultiChannelBufferView`] (non-owning).
//! - [`ChannelLayout`]: Struct describing the active channels of a connection between nodes. Graph edges
//!   can carry an optional channel layout to indicate which channels of a node's output buffer are
//!   processed by connected successor nodes.
//! - [`ProcessingContext`]: Struct providing context for audio processing, including input and output buffer
//!   references, the channel layout, and the number of frames to process.
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
//! - [`DspGraph::add_processor`]: Adds a new processor node to the graph along with its associated output buffer.
//! - [`DspGraph::connect`]: Connects two nodes in the graph with an edge, optionally specifying a channel layout.
//!
//! Additional methods are provided for more advanced operations, such as enabling and
//! disabling edges and removing connections.
//!
//! For channel rewiring support, see [`RewireDspGraph`], which extends the base graph with
//! the ability to remap channels on existing connections.
//!
//! Graph nodes and edges are identified using the `NodeIndex` and `EdgeIndex` types from the `petgraph` crate,
//! which provides the underlying directed graph implementation.
//!
//! Input and output nodes: These are special nodes that serve as the entry and exit points of the graph.
//! Input and output nodes do not process any audio data and do not have dedicated buffers within the graph
//! structure. The corresponding buffers are passed to the [`DspGraph::process`] method instead. See also [`GraphNode`]
//! and [`DspGraph::connect`].
//!
//! # Realtime safety
//!
//! Realtime safety is guaranteed for all [`BasicDspGraph`] operations, including processing and modifying
//! the graph structure. [`RewireDspGraph`] additionally supports channel rewiring, which is not
//! realtime-safe and is documented as such.
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

/// Identifier for a graph node
///
/// Input and Output nodes are special nodes representing the entry and exit points of the graph.
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
#[doc(hidden)]
pub struct EdgeData {
    pub channel_layout: Option<ChannelLayout>,
    pub enabled: bool,
}

/// Common interface for graph edge types
#[doc(hidden)]
pub trait GraphEdge: Clone {
    const SUPPORTS_REWIRE: bool; // for compile-time differentiation between edge types

    fn new(channel_layout: Option<ChannelLayout>) -> Self;
    fn data(&self) -> &EdgeData;
    fn data_mut(&mut self) -> &mut EdgeData;

    fn get_rewire(&self) -> Option<&HashMap<usize, usize>> {
        None
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct ProcessorChannel {
    pub data: EdgeData,
}

impl GraphEdge for ProcessorChannel {
    const SUPPORTS_REWIRE: bool = false;

    fn new(channel_layout: Option<ChannelLayout>) -> Self {
        Self {
            data: EdgeData {
                channel_layout,
                enabled: true,
            },
        }
    }

    fn data(&self) -> &EdgeData {
        &self.data
    }

    fn data_mut(&mut self) -> &mut EdgeData {
        &mut self.data
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct RewireProcessorChannel {
    pub data: EdgeData,
    pub rewire: Option<HashMap<usize, usize>>,
}

impl GraphEdge for RewireProcessorChannel {
    const SUPPORTS_REWIRE: bool = true;

    fn new(channel_layout: Option<ChannelLayout>) -> Self {
        Self {
            data: EdgeData {
                channel_layout,
                enabled: true,
            },
            rewire: None,
        }
    }

    fn data(&self) -> &EdgeData {
        &self.data
    }

    fn data_mut(&mut self) -> &mut EdgeData {
        &mut self.data
    }

    fn get_rewire(&self) -> Option<&HashMap<usize, usize>> {
        self.rewire.as_ref()
    }
}

type GraphVisitMap<T, E> = <StableDiGraph<ProcessorNode<T>, E> as petgraph::visit::Visitable>::Map;

/// Directed graph structure for audio processing
///
/// Consists of processor nodes and typed edges that describe the signal flow.
/// When constructing a [`DspGraph`], a capacity needs to be provided to preallocate internal data
/// structures. This ensures realtime safety when adding nodes and edges at runtime.
///
/// Two graph variants are provided by the library, differentiated by their edge types:
/// - [`BasicDspGraph`]: lightweight graph without rewiring support
/// - [`RewireDspGraph`]: graph with channel rewiring capabilities
///
/// Most users should use the type aliases rather than `DspGraph` directly:
///
/// Nodes and edges can be added using [`DspGraph::add_processor`] and [`DspGraph::connect`].
/// Connections can also be dynamically enabled, disabled or updated.
///
/// The graph owns the processing nodes as well as their corresponding output buffers. Conversely,
/// the input and output buffers that the graph operates on must be managed outside of the graph structure.
/// To run the audio processing graph, use the [`DspGraph::process`] method, which takes the input
/// and output buffers as arguments.
#[allow(private_bounds)]
pub struct DspGraph<T: Sample, Edge: GraphEdge> {
    graph: StableDiGraph<ProcessorNode<T>, Edge>,
    topo_order: Vec<NodeIndex>, // Pre-allocated processing order vector
    buffers: Vec<Option<MultiChannelBuffer<T>>>,
    input_node: NodeIndex,
    output_node: NodeIndex,
    summing_buffer: MultiChannelBuffer<T>,
    dfs_visitor: DfsPostOrder<NodeIndex, GraphVisitMap<T, Edge>>,
    topo_dirty: bool,
}

#[allow(private_bounds)]
impl<T: Sample, Edge: GraphEdge> DspGraph<T, Edge> {
    /// Creates a new [`DspGraph`] with preallocated capacity for internal data structures
    ///
    /// - `num_channels`: Maximum number of channels a node will process. Needs to be known upfront for
    ///   summing operations.
    /// - `frame_size`: Maximum number of frames for block-wise processing.
    /// - `max_num_edges`: Graph capacity used for preallocation. This value also bounds the maximum
    ///   number of nodes that can be added. If `None`, defaults to 64.
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

    /// Adds a processor node to the graph along with its associated output buffer
    ///
    /// Returns the `NodeIndex` of the newly added processor node (or an error if something went wrong).
    /// The node index can be used to reference the node when adding an edge to it using [`DspGraph::connect`].
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

    /// Connects two nodes in the graph with an edge
    ///
    /// The direction of the edge is so that the `to` node will access the output buffer of the `from` node
    /// as its input buffer during processing. These nodes can be either processor nodes added via
    /// [`DspGraph::add_processor`], or the input or output nodes of the entire graph (see [`GraphNode`]).
    ///
    /// Optionally, a [`ChannelLayout`] can be provided to specify which channels of the `from` node's output buffer
    /// should be processed by the `to` node. If no channel layout is provided, all channels are connected by default.
    ///
    /// Returns the `EdgeIndex` of the newly created edge (or an error if something went wrong).
    /// The edge index can be used to reference the edge for further operations such as enabling/disabling
    /// the connection.
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

        let edge = Edge::new(channel_layout);

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

    /// Removes an existing connection from the graph
    ///
    /// **NOT realtime safe for the [`RewireDspGraph`] variant**
    ///
    /// Alternative: Use [`DspGraph::disable_connection`] to temporarily disable a connection.
    pub fn remove_connection(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        self.graph.remove_edge(edge).ok_or("Edge not found")?;
        self.topo_dirty = true;
        Ok(())
    }

    /// Enables an existing connection in the graph
    pub fn enable_connection(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        if let Some(edge_weight) = self.graph.edge_weight_mut(edge) {
            edge_weight.data_mut().enabled = true;
            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    /// Disables an existing connection in the graph
    pub fn disable_connection(&mut self, edge: EdgeIndex) -> Result<(), AudioGraphError> {
        if let Some(edge_weight) = self.graph.edge_weight_mut(edge) {
            edge_weight.data_mut().enabled = false;
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

    /// Processes audio data through the graph
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
                .filter(|edge| edge.weight().data().enabled);

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

                    let edge_layout = edge.weight().data().channel_layout.clone();

                    if Edge::SUPPORTS_REWIRE {
                        if let Some(rewire) = edge.weight().get_rewire() {
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
                let (input_node, channel_layout, rewire_map) = {
                    let edge = incoming_edges.next().unwrap();
                    let rewire = if Edge::SUPPORTS_REWIRE {
                        edge.weight().get_rewire().cloned()
                    } else {
                        None
                    };
                    (
                        edge.source(),
                        edge.weight().data().channel_layout.clone(),
                        rewire,
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

                if Edge::SUPPORTS_REWIRE {
                    if let Some(rewire) = &rewire_map {
                        let rewired_buffer_view = RewiredBufferView {
                            buffer: input_buffer,
                            rewire,
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
            .filter(|e| e.weight().data().enabled)
        {
            let node = edge.source();
            let node_buffer: &dyn AudioBuffer<T> = if node == self.input_node {
                input
            } else {
                let input_buffer_index = node.index();
                self.buffers[input_buffer_index].as_ref().unwrap()
            };
            // TODO: handle disconnected channels
            output.add(node_buffer, &edge.weight().data().channel_layout.clone());
        }
    }
}

#[allow(private_bounds)]
impl<T: Sample> RewireDspGraph<T> {
    /// Rewires an existing connection in the graph to use a different channel mapping
    /// between the edge's source and destination nodes.
    ///
    /// **NOT realtime safe**
    ///
    /// The `rewire_mapping` parameter is a slice of tuples where each tuple defines a channel mapping
    /// in the form `(source_channel, destination_channel)`.
    ///
    /// Note: Mapping multiple source channels to the same destination channel returns an error.
    ///
    /// # Example
    /// ```rust
    /// use audiograph::{FrameSize, GraphNode, MultiChannelBuffer, NoOp, RewireDspGraph};
    ///
    /// let frame_size = FrameSize(1024);
    ///
    /// let mut dsp_graph = RewireDspGraph::<f32>::new(4, frame_size, None);
    ///
    /// let node1 = dsp_graph
    ///     .add_processor(
    ///         NoOp {},
    ///         MultiChannelBuffer::new(4, frame_size), // 4 output channels
    ///     )
    ///     .unwrap();
    ///
    /// let node2 = dsp_graph
    ///     .add_processor(
    ///         NoOp {},
    ///         MultiChannelBuffer::new(4, frame_size), // 4 output channels
    ///     )
    ///     .unwrap();
    ///
    /// // Connect nodes with default channel layout (i.e., all channels in order)
    /// dsp_graph
    ///     .connect(GraphNode::Input, node1.into(), None)
    ///     .unwrap();
    ///
    /// let edge = dsp_graph.connect(node1.into(), node2.into(), None).unwrap();
    ///
    /// dsp_graph
    ///     .connect(node2.into(), GraphNode::Output, None)
    ///     .unwrap();
    ///
    /// // Rewire the edge to swap channels 0 and 1, while keeping channels 2 and 3 the same
    /// dsp_graph
    ///     .rewire(edge, &[(0, 1), (1, 0), (2, 2), (3, 3)])
    ///     .unwrap();
    /// ```
    ///
    pub fn rewire(
        &mut self,
        edge_index: EdgeIndex,
        rewire_mapping: &[(usize, usize)], // maps source channel to destination channel
    ) -> Result<(), AudioGraphError> {
        if let Some(edge) = self.graph.edge_weight_mut(edge_index) {
            let mut channel_layout = ChannelLayout::new(0);
            let mut rewire = HashMap::new();

            for &(source, dest) in rewire_mapping {
                channel_layout.connect(dest);

                // we flip (source, dest) to have logical to physical mapping
                if rewire.insert(dest, source).is_some() {
                    return Err("Duplicate destination channel in rewire mapping");
                }
            }

            edge.data_mut().channel_layout = Some(channel_layout);
            edge.rewire = Some(rewire);

            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    /// Removes rewiring from an existing connection
    ///
    /// **NOT realtime-safe**
    ///
    /// TODO: a valid channel layout is not established after removal!
    pub fn remove_rewire(&mut self, edge_index: EdgeIndex) -> Result<(), AudioGraphError> {
        if let Some(edge) = self.graph.edge_weight_mut(edge_index) {
            edge.rewire = None;
            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    /// Connects two nodes and creates a rewired mapping in one step
    ///
    /// **NOT realtime-safe**
    ///
    /// See [`RewireDspGraph::connect`] and [`RewireDspGraph::rewire`] for details.
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
}

/// Type alias for a basic graph without rewiring support
#[allow(private_interfaces)]
pub type BasicDspGraph<T> = DspGraph<T, ProcessorChannel>;

/// Type alias for a graph with rewiring support
#[allow(private_interfaces)]
pub type RewireDspGraph<T> = DspGraph<T, RewireProcessorChannel>;

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
        let mut graph = BasicDspGraph::<f32>::new(1, FrameSize(10), None);
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
        let mut graph = BasicDspGraph::<f32>::new(1, FrameSize(10), None);
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
        let mut graph = BasicDspGraph::<f32>::new(1, FrameSize(10), None);
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
        let mut graph = BasicDspGraph::<f32>::new(3, FrameSize(10), None);
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

        let mut graph = BasicDspGraph::<f32>::new(1, FrameSize(8), None);

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

        let mut graph = BasicDspGraph::<f32>::new(2, frame_size, Some(16));

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
        let mut graph = BasicDspGraph::<f32>::new(1, FrameSize(10), Some(4));

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
        let mut graph = BasicDspGraph::<f32>::new(1, frame_size, None);

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
