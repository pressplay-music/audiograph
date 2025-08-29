use audiograph::{
    buffer::{AudioBuffer, FrameSize, MultiChannelBuffer},
    processor::{ProcessingContext, Processor},
    DspGraph, GraphNode,
};
use clap::Parser;
use cpal::traits::*;
use std::time::Duration;

#[derive(Parser)]
struct Args {
    /// Output device name
    device: String,

    /// Frequency in Hz
    frequency: f32,

    /// Gain (0.0 to 1.0)
    gain: f32,
}

struct SineWaveGen {
    phase: f32,
    sine_buffer: Vec<f32>,
    phase_increment: f32,
}

impl SineWaveGen {
    fn new(frequency: f32, sample_rate: f32, frame_size: FrameSize) -> Self {
        Self {
            phase: 0.0,
            sine_buffer: vec![0.0; frame_size.0],
            phase_increment: 2.0 * std::f32::consts::PI * frequency / sample_rate,
        }
    }
}

impl Processor<f32> for SineWaveGen {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        for sample in self.sine_buffer.iter_mut() {
            *sample = self.phase.sin();
            self.phase = (self.phase + self.phase_increment) % (2.0 * std::f32::consts::PI);
        }

        for channel_idx in 0..context.output_buffer.num_channels() {
            let output_channel = context.output_buffer.channel_mut(channel_idx).unwrap();
            output_channel.copy_from_slice(&self.sine_buffer);
        }
    }
}

struct Gain {
    gain: f32,
}

impl Gain {
    fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl Processor<f32> for Gain {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        for channel in context.channel_layout.iter() {
            let input_channel = context.input_buffer.channel(channel).unwrap();
            let output_channel = context.output_buffer.channel_mut(channel).unwrap();
            for (input, output) in input_channel.iter().zip(output_channel.iter_mut()) {
                *output = input * self.gain;
            }
        }
    }
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

                dsp_graph.process(&input_buffer, &mut output_buffer);

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
