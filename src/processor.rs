use crate::{buffer::AudioBuffer, channel::ChannelLayout, sample::Sample};

pub trait Processor<T: Sample> {
    fn process(&mut self, context: &mut ProcessingContext<T>);
}

pub struct ProcessingContext<'a, T: Sample> {
    pub input_buffer: &'a dyn AudioBuffer<T>,
    pub output_buffer: &'a mut dyn AudioBuffer<T>,
    pub channel_layout: ChannelLayout,
}

pub struct PassThrough;

impl<T: Sample> Processor<T> for PassThrough {
    fn process(&mut self, context: &mut ProcessingContext<T>) {
        for channel in context.channel_layout.iter() {
            let input_channel = context.input_buffer.channel(channel).unwrap();
            let output_channel = context.output_buffer.channel_mut(channel).unwrap();
            output_channel.copy_from_slice(input_channel);
        }
    }
}

pub struct NoOp;

impl<T: Sample> Processor<T> for NoOp {
    fn process(&mut self, _context: &mut ProcessingContext<T>) {}
}
