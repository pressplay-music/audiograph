use audiograph::{
    DspGraph, GraphNode,
    buffer::{AudioBuffer, FrameSize, MultiChannelBuffer},
};
use clap::Parser;
use cpal::traits::*;
use std::time::Duration;

mod processors;
use processors::{Gain, SineWaveGen};

#[derive(Parser)]
struct Args {
    /// Output device name
    device: String,

    /// Frequency in Hz
    frequency: f32,

    /// Gain (0.0 to 1.0)
    gain: f32,
}

fn main() {
    let args = Args::parse();

    println!("Device: {}", args.device);
    println!("Frequency: {} Hz", args.frequency);
    println!("Gain: {}", args.gain);

    let host = cpal::default_host();

    let device = host
        .output_devices()
        .unwrap()
        .find(|device| {
            device
                .name()
                .map(|device_name| device_name.contains(&args.device))
                .unwrap_or(false)
        })
        .expect("Failed to find output device");

    let mut config = device.default_output_config().unwrap().config();
    config.buffer_size = cpal::BufferSize::Fixed(512);

    println!("Config: {:?}", config);

    let frame_size = FrameSize(512);
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    // -- CREATE GRAPH AND ADD NODES -- //

    let mut dsp_graph = DspGraph::<f32>::new(channels, frame_size, Some(8));

    let sine_node = dsp_graph
        .add_processor(
            SineWaveGen::new(args.frequency, sample_rate, frame_size),
            MultiChannelBuffer::new(channels, frame_size),
        )
        .unwrap();

    let gain_node = dsp_graph
        .add_processor(
            Gain::new(args.gain),
            MultiChannelBuffer::new(channels, frame_size),
        )
        .unwrap();

    // -- CONNECT NODES --

    dsp_graph
        .connect(GraphNode::Input, sine_node.into(), None)
        .unwrap();
    dsp_graph
        .connect(sine_node.into(), gain_node.into(), None)
        .unwrap();
    dsp_graph
        .connect(gain_node.into(), GraphNode::Output, None)
        .unwrap();

    let input_buffer = MultiChannelBuffer::<f32>::new(channels, frame_size); // not used by the existing graph nodes, but required by API
    let mut output_buffer = MultiChannelBuffer::<f32>::new(channels, frame_size);

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let cpal_frames = data.len() / channels;

                if cpal_frames != frame_size.0 {
                    eprintln!(
                        "Buffer size mismatch: expected {}, got {}",
                        frame_size.0, cpal_frames
                    );
                    data.fill(0.0);
                    return;
                }

                dsp_graph.process(&input_buffer, &mut output_buffer, frame_size);

                output_buffer.copy_to_interleaved(data);
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )
        .expect("Failed to build stream");

    stream.play().expect("Failed to start stream");

    println!("Playing for 4 seconds...");
    std::thread::sleep(Duration::from_secs(4));
    println!("Done!");
}
