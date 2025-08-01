use crate::{
    buffer::{iter_channels, iter_channels_mut, AudioBuffer},
    channel::ChannelLayout,
    sample::Sample,
};

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
    // TODO: find way to implement channel iterators
    fn process(&mut self, context: &mut ProcessingContext<T>) {
        for (input_channel, output_channel) in
            iter_channels(context.input_buffer).zip(iter_channels_mut(context.output_buffer))
        {
            output_channel.copy_from_slice(input_channel);
        }
    }
}

pub struct NoOp;

impl<T: Sample> Processor<T> for NoOp {
    fn process(&mut self, _context: &mut ProcessingContext<T>) {}
}
