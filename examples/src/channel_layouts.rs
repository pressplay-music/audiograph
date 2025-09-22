use audiograph::{
    buffer::{AudioBuffer, FrameSize, MultiChannelBuffer},
    channel::ChannelLayout,
    DspGraph, GraphNode,
};
use clap::Parser;

mod processors;
use processors::{Gain, SineWaveGen};

#[derive(Parser)]
struct Args {
    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right)
    pan: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let frame_size = FrameSize(16);

    let mut dsp_graph = DspGraph::<f32>::new(2, frame_size, Some(8));

    let sine_node = dsp_graph
        .add_processor(
            SineWaveGen::new(10.0, 100.0, frame_size),
            MultiChannelBuffer::new(2, frame_size),
        )
        .unwrap();

    let left_gain_node = dsp_graph
        .add_processor(
            Gain::new(1.0 - args.pan.max(0.0)),
            MultiChannelBuffer::new(1, frame_size), // only one output channel
        )
        .unwrap();

    let right_gain_node = dsp_graph
        .add_processor(
            Gain::new(1.0 + args.pan.min(0.0)),
            MultiChannelBuffer::new(1, frame_size), // only one output channel
        )
        .unwrap();

    let output_gain_node = dsp_graph
        .add_processor(Gain::new(0.5), MultiChannelBuffer::new(2, frame_size))
        .unwrap();

    dsp_graph.connect(
        GraphNode::Input,
        sine_node.into(),
        Some(ChannelLayout::new(2)),
    )?;

    dsp_graph.connect(
        sine_node.into(),
        left_gain_node.into(),
        Some(ChannelLayout::from_indices(&[0])), // left channel
    )?;

    dsp_graph.connect(
        sine_node.into(),
        right_gain_node.into(),
        Some(ChannelLayout::from_indices(&[1])), // right channel
    )?;

    dsp_graph.connect(
        left_gain_node.into(),
        output_gain_node.into(),
        Some(ChannelLayout::from_indices(&[0])), // left channel
    )?;

    dsp_graph.connect(
        right_gain_node.into(),
        output_gain_node.into(),
        Some(ChannelLayout::from_indices(&[1])),
    )?;

    dsp_graph.connect(
        output_gain_node.into(),
        GraphNode::Output,
        Some(ChannelLayout::new(2)),
    )?;

    let input_buffer = MultiChannelBuffer::<f32>::new(2, frame_size);
    let mut output_buffer = MultiChannelBuffer::<f32>::new(2, frame_size);

    for _ in 0..10 {
        dsp_graph.process(&input_buffer, &mut output_buffer);
        println!("{:?}", output_buffer.channel(0).unwrap());
        println!("{:?}", output_buffer.channel(1).unwrap());
    }

    Ok(())
}
