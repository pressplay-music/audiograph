use crate::{
    buffer::{AudioBuffer, FrameSize},
    channel::ChannelLayout,
    sample::Sample,
};

pub trait Processor<T: Sample> {
    fn process(&mut self, context: &mut ProcessingContext<T>);
}

#[non_exhaustive]
pub struct ProcessingContext<'a, T: Sample> {
    pub input_buffer: &'a dyn AudioBuffer<T>,
    pub output_buffer: &'a mut dyn AudioBuffer<T>,
    pub channel_layout: ChannelLayout,
    pub num_frames: FrameSize,
}

impl<'a, T: Sample> ProcessingContext<'a, T> {
    pub fn create_unchecked(
        input_buffer: &'a dyn AudioBuffer<T>,
        output_buffer: &'a mut dyn AudioBuffer<T>,
        channel_layout: ChannelLayout,
        num_frames: FrameSize,
    ) -> Self {
        Self {
            input_buffer,
            output_buffer,
            channel_layout,
            num_frames,
        }
    }

    pub fn create_checked(
        input_buffer: &'a dyn AudioBuffer<T>,
        output_buffer: &'a mut dyn AudioBuffer<T>,
        mut channel_layout: ChannelLayout,
        num_frames: FrameSize,
    ) -> Self {
        let max_channels = input_buffer
            .num_channels()
            .min(output_buffer.num_channels());
        channel_layout.clamp(max_channels);

        let max_frames = input_buffer
            .num_frames()
            .0
            .min(output_buffer.num_frames().0);
        let num_frames = FrameSize(num_frames.0.min(max_frames));

        Self {
            input_buffer,
            output_buffer,
            channel_layout,
            num_frames,
        }
    }
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
