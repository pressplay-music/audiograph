use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

pub struct MultiChannelBuffer {}

impl MultiChannelBuffer {
    pub fn clear(&mut self) {
        // Clear all channels
    }

    pub fn add(&mut self, _other: &MultiChannelBuffer) {
        // Add other buffer to self
        println!("Adding buffer");
    }
}

pub trait Processor {
    fn process(
        &mut self,
        input_buffer: &MultiChannelBuffer,
        output_buffer: &mut MultiChannelBuffer,
    );
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
    ) {
        self.processor.process(input_buffer, output_buffer);
    }
}

pub struct AudioEdge {}
impl AudioEdge {
    pub fn edge_info(&self) {
        // Get some content
    }
}

pub struct DspGraph {
    graph: DiGraph<ProcessorNode, AudioEdge>,
    topo_order: Vec<NodeIndex>, // Precomputed processing order
    buffers: Vec<MultiChannelBuffer>,
    buffer_map: HashMap<NodeIndex, usize>,
    summing_buffer: MultiChannelBuffer,
}

impl DspGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            topo_order: Vec::new(),
            buffers: Vec::new(),
            buffer_map: HashMap::new(),
            summing_buffer: MultiChannelBuffer {},
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
    pub fn process(&mut self, input: &MultiChannelBuffer, _output: &mut MultiChannelBuffer) {
        self.summing_buffer.clear();

        // Process the first node
        let entry_node_index = self.topo_order[0];
        let entry_node = self.graph.node_weight_mut(entry_node_index).unwrap();
        let output_buffer = self
            .buffer_map
            .get(&entry_node_index)
            .and_then(|&i| self.buffers.get_mut(i))
            .unwrap();
        entry_node.process(input, output_buffer);

        // Process remaining nodes
        for &node_index in self.topo_order.iter().skip(1) {
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
                processor_node
                    .processor
                    .process(&self.summing_buffer, output_buffer);
            } else {
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
                let _info = edge.weight().edge_info();

                let processor_node = self.graph.node_weight_mut(node_index).unwrap();
                processor_node
                    .processor
                    .process(input_buffer, output_buffer);
            }
        }

        // TODO: Copy output buffer to provided output buffer
        // TODO: optimize so that no copies are needed on output node
    }
}

struct FooProcessor {}
struct BarProcessor {}
struct BazProcessor {}

impl Processor for FooProcessor {
    fn process(
        &mut self,
        _input_buffer: &MultiChannelBuffer,
        _output_buffer: &mut MultiChannelBuffer,
    ) {
        println!("FooProcessor processing");
    }
}

impl Processor for BarProcessor {
    fn process(
        &mut self,
        _input_buffer: &MultiChannelBuffer,
        _output_buffer: &mut MultiChannelBuffer,
    ) {
        println!("BarProcessor processing");
    }
}

impl Processor for BazProcessor {
    fn process(
        &mut self,
        _input_buffer: &MultiChannelBuffer,
        _output_buffer: &mut MultiChannelBuffer,
    ) {
        println!("BazProcessor processing");
    }
}

fn main() {
    let mut graph = DspGraph::new();

    let foo = graph.add_processor(FooProcessor {}, MultiChannelBuffer {});
    let bar1 = graph.add_processor(BarProcessor {}, MultiChannelBuffer {});
    graph.connect(foo, bar1);

    let baz = graph.add_processor(BazProcessor {}, MultiChannelBuffer {});
    graph.connect(bar1, baz);

    let bar2 = graph.add_processor(BarProcessor {}, MultiChannelBuffer {});
    graph.connect(foo, bar2);
    graph.connect(bar2, baz);

    let bar3 = graph.add_processor(BarProcessor {}, MultiChannelBuffer {});
    graph.connect(foo, bar3);
    graph.connect(bar3, baz);

    let input = MultiChannelBuffer {};
    let mut output = MultiChannelBuffer {};

    graph.process(&input, &mut output);
}
