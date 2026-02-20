use audiograph::{
    GraphNode, RewireDspGraph,
    buffer::{AudioBuffer, FrameSize, MultiChannelBuffer},
    channel::ChannelSelection,
};
use clap::Parser;

mod processors;
use processors::{Gain, SineWaveGen};

#[derive(Parser)]
struct Args {
    /// Pan position (0.0 = left, 0.5 = center, 1.0 = right)
    pan: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let frame_size = FrameSize(4);

    let mut dsp_graph = RewireDspGraph::<f32>::new(2, frame_size, Some(8));

    let sine_node = dsp_graph
        .add_processor(
            SineWaveGen::new(10.0, 100.0, frame_size),
            MultiChannelBuffer::new(2, frame_size),
        )
        .unwrap();

    let left_gain_node = dsp_graph
        .add_processor(
            Gain::new(1.0 - args.pan),
            MultiChannelBuffer::new(1, frame_size), // only one output channel
        )
        .unwrap();

    let right_gain_node = dsp_graph
        .add_processor(
            Gain::new(args.pan),
            MultiChannelBuffer::new(1, frame_size), // only one output channel
        )
        .unwrap();

    let output_gain_node = dsp_graph
        .add_processor(Gain::new(0.5), MultiChannelBuffer::new(2, frame_size))
        .unwrap();

    dsp_graph.connect(
        GraphNode::Input,
        sine_node.into(),
        Some(ChannelSelection::new(2)),
    )?;

    dsp_graph.connect(
        sine_node.into(),
        left_gain_node.into(),
        Some(ChannelSelection::new(1)), // first channel only (left)
    )?;

    // Connect to right gain node and rewire the edge to use channel 1 instead of 0
    dsp_graph.connect_rewired(sine_node.into(), right_gain_node.into(), &[(1, 0)])?; // second channel only (first channel after rewire)

    dsp_graph.connect(
        left_gain_node.into(),
        output_gain_node.into(),
        Some(ChannelSelection::new(1)), // left gain node has only one output channel
    )?;

    let right_to_output_edge = dsp_graph.connect(
        right_gain_node.into(),
        output_gain_node.into(),
        Some(ChannelSelection::new(1)),
    )?;

    // Rewire existing connection: channel 0 of right node to channel 1 of output node (could use connect_rewired directly)
    dsp_graph.rewire(right_to_output_edge, &[(0, 1)])?;

    dsp_graph.connect(
        output_gain_node.into(),
        GraphNode::Output,
        Some(ChannelSelection::new(2)),
    )?;

    let input_buffer = MultiChannelBuffer::<f32>::new(2, frame_size);
    let mut output_buffer = MultiChannelBuffer::<f32>::new(2, frame_size);

    for _ in 0..8 {
        dsp_graph.process(&input_buffer, &mut output_buffer, frame_size);
        println!("L: {:?}", output_buffer.channel(0).unwrap());
        println!("R: {:?}", output_buffer.channel(1).unwrap());
    }

    Ok(())
}
