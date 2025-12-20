use audiograph::{
    buffer::FrameSize,
    processor::{ProcessingContext, Processor},
};

pub struct Gain {
    gain: f32,
}

impl Gain {
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl Processor<f32> for Gain {
    fn process(&mut self, context: &mut ProcessingContext<f32>) {
        context.for_each_channel(|input_channel, output_channel| {
            for (input, output) in input_channel.iter().zip(output_channel.iter_mut()) {
                *output = input * self.gain;
            }
        });
    }
}

pub struct SineWaveGen {
    phase: f32,
    sine_buffer: Vec<f32>,
    phase_increment: f32,
}

impl SineWaveGen {
    pub fn new(frequency: f32, sample_rate: f32, frame_size: FrameSize) -> Self {
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
