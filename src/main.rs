use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

pub struct MultiChannelBuffer {}

impl MultiChannelBuffer {
    pub fn clear(&mut self) {
        // Clear all channels
    }

    pub fn add(&mut self, other: &MultiChannelBuffer) {
        // Add other buffer to self
    }
}

pub trait Processor {
    fn process(
        &mut self,
        input_buffer: &MultiChannelBuffer,
        output_buffer: &mut MultiChannelBuffer,
    );
}

/// Represents an audio processing node with its preallocated output buffer.
pub struct ProcessorNode {
    processor: Box<dyn Processor + Send>,
    output_buffer: MultiChannelBuffer, // Each node owns its output buffer
}

impl ProcessorNode {
    pub fn new<T: Processor + Send + 'static>(
        processor: T,
        output_buffer: MultiChannelBuffer,
    ) -> Self {
        Self {
            processor: Box::new(processor),
            output_buffer,
        }
    }

    pub fn process(&mut self, input_buffer: &MultiChannelBuffer) {
        self.processor
            .process(input_buffer, &mut self.output_buffer);
    }
}

pub struct AudioEdge {}

/// The DSP graph managing processors and audio flow.
pub struct DspGraph {
    graph: DiGraph<ProcessorNode, AudioEdge>,
    topo_order: Vec<NodeIndex>, // Precomputed processing order
    summing_buffer: MultiChannelBuffer,
}

impl DspGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            topo_order: Vec::new(),
            summing_buffer: MultiChannelBuffer {},
        }
    }

    pub fn add_processor<T: Processor + Send + 'static>(
        &mut self,
        processor: T,
        output_buffer: MultiChannelBuffer,
    ) -> NodeIndex {
        self.graph
            .add_node(ProcessorNode::new(processor, output_buffer))
    }

    pub fn connect(&mut self, from: NodeIndex, to: NodeIndex) {
        let edge = AudioEdge {};
        self.graph.add_edge(from, to, edge);

        // Precompute topological order for real-time safety
        self.recompute_topo_order();
    }

    fn recompute_topo_order(&mut self) {
        // TODO: propagate graph error (e.g. cycles)
        self.topo_order = petgraph::algo::toposort(&self.graph, None)
            .expect("Graph has cycles! Ensure DAG structure.");
    }

    /// Processes the graph using provided input and output buffers.
    pub fn process(&mut self, input: &MultiChannelBuffer, output: &mut MultiChannelBuffer) {
        self.summing_buffer.clear();

        let entry_node = self.graph.node_weight_mut(self.topo_order[0]).unwrap();
        entry_node.process(input);
        drop(entry_node);

        // skip first and last node
        for node_index in self
            .topo_order
            .iter()
            .skip(1)
            .take(self.topo_order.len() - 2)
        {
            let num_incoming_edges = self
                .graph
                .edges_directed(*node_index, Direction::Incoming)
                .count();

            // are there multiple inputs? if so, we need to sum them
            if num_incoming_edges > 1 {
                self.summing_buffer.clear();

                self.graph
                    .edges_directed(*node_index, Direction::Incoming)
                    .for_each(|edge| {
                        let input_node = edge.source();
                        let input_buffer =
                            &self.graph.node_weight(input_node).unwrap().output_buffer;
                        self.summing_buffer.add(input_buffer);
                    });

                let ProcessorNode {
                    processor,
                    output_buffer,
                } = self.graph.node_weight_mut(*node_index).unwrap();

                processor.process(&self.summing_buffer, output_buffer);
            } else {
                let input_node = self
                    .graph
                    .neighbors_directed(*node_index, Direction::Incoming)
                    .next()
                    .unwrap();

                let input_buffer = &self.graph.node_weight(input_node).unwrap().output_buffer;

                let ProcessorNode {
                    processor,
                    output_buffer,
                } = self.graph.node_weight_mut(*node_index).unwrap();

                processor.process(input_buffer, output_buffer);
            }
        }
    }
}
fn main() {
    println!("Hello, world!");
}
