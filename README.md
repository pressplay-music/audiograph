# Audiograph

**A realtime audio processing graph library for Rust**

Audiograph provides abstractions for audio processors, audio buffers and channel routing, and enables the construction and management of directed signal processing graphs. Graph edges can explicitly encode the channel-based routing between processor nodes, allowing for flexible channel selection and reordering with minimal runtime overhead.

Audiograph supports modifying DSP graphs at runtime under real-time constraints, including adding or removing processor nodes, and changing connections between nodes.

## Basic Usage

The most basic functionality is shown below. For more advanced usage, refer to the documentation.

### Create a graph instance

```rust
use audiograph::{DspGraph, FrameSize};

let num_channels = 2; // stereo
let frame_size = FrameSize(256); // maximum number of samples per frame

// intended maximum number of edges in the graph,
// used for the preallocation of internal graph structures
let max_num_edges = Some(20);

// we use f32 as the sample type
let mut dsp_graph = DspGraph::<f32>::new(num_channels, frame_size, max_num_edges);
```

### Define a processing node

You have to implement the `Processor` trait for your custom processor types.

```rust
use audiograph::{ProcessingContext, Processor, Sample};

struct MyProcessor;

// We can implement for a generic (floating point) sample type T
impl<T: Sample> Processor<T> for MyProcessor {
    fn process(&mut self, context: &mut ProcessingContext<T>) {
        // the ProcessingContext references the input and output buffers
        // and includes additional information for processing
    }
}
```

### Add processor nodes to the graph

```rust
use audiograph::MultiChannelBuffer;

// allocate a sample buffer that the processor node can write into
let node_buffer = MultiChannelBuffer::new(num_channels, frame_size);

let my_processor_node = dsp_graph.add_processor(MyProcessor {}, node_buffer).unwrap();
```

### Connect nodes in the graph

```rust
use audiograph::GraphNode;

// connect graph input to a processor node
dsp_graph
    .connect(GraphNode::Input, my_processor_node.into(), None)
    .unwrap();

// connect two processor nodes
dsp_graph
    .connect(my_processor_node.into(), some_other_node.into(), None)
    .unwrap();

// connect a processor node to the graph output
dsp_graph
    .connect(some_other_node.into(), GraphNode::Output, None)
    .unwrap();
```

### Execute the graph

```rust
// The input and output buffers may come from an audio I/O callback.
// There are buffer view wrappers that can be used in this case.
// Or you define and store your own MultiChannelBuffer instances instead:

let input_buffer = MultiChannelBuffer::new(channels, frame_size);
let mut output_buffer = MultiChannelBuffer::new(channels, frame_size);

// Do the processing
dsp_graph.process(&input_buffer, &mut output_buffer, frame_size);
```

## Technical Notes

- Audiograph operates on channel-based floating-point audio sample structures. Convenience utilities are provided for interleaving and deinterleaving buffers when interfacing with APIs that use interleaved layouts.

- Realtime safety is guaranteed for most of the graph operations, including processing and modifying the graph structure. Some methods are not realtime safe (currently e.g. the `rewire` method). These methods are documented as such. Some operations require audio buffers to be preallocated (such as adding nodes to the graph), which is the responsibility of the user of the library.

- There is a maximum number of channels the `DspGraph` can process. The default value is 64, but it can be changed at compile time by setting the `MAX_CHANNELS` environment variable when building the library. It is not worth to make this value smaller. Higher values will introduce slightly more processing overhead.

- The acyclic directed graph structure is implemented using the `petgraph` crate (https://github.com/petgraph/petgraph).

### A note on performance

The graph is designed to be efficient, although that is not its primary goal. There are minor performance penalties stemming from branching, lookups, and dynamic dispatching. For typical use cases, most of the CPU time is likely spent on the audio processing code itself, with minimal overhead from graph management.

Iterating over a `ChannelLayout` can be a bit costly because this struct is organized as a bit set.

If you have very high performance requirements, consider not using a graph structure at all.

## Alternatives

The `dasp_graph` module of the `DASP` crate (https://github.com/RustAudio/dasp/tree/master/dasp_graph) provides an alternative audio graph implementation in Rust, also built on top of `petgraph`. It is more generic and lower-level, with certain structural constraints.

Audiograph places more responsibility on the graph itself to express e.g. complex routing, allowing processing nodes to remain context-agnostic. This design introduces slightly higher graph overhead.

## TODO

- [ ] Make rewiring of edges realtime-safe (e.g. by using fixed-size arrays for channel mappings)
