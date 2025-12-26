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
    pub channel_layout: Option<ChannelLayout>,
    pub num_frames: FrameSize,
}

impl<'a, T: Sample> ProcessingContext<'a, T> {
    pub fn create_unchecked(
        input_buffer: &'a dyn AudioBuffer<T>,
        output_buffer: &'a mut dyn AudioBuffer<T>,
        channel_layout: Option<ChannelLayout>,
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
        mut num_frames: FrameSize,
    ) -> Self {
        let max_channels = input_buffer
            .num_channels()
            .min(output_buffer.num_channels());
        channel_layout.clamp(max_channels);

        let max_frames = input_buffer
            .num_frames()
            .0
            .min(output_buffer.num_frames().0);
        num_frames = FrameSize(num_frames.0.min(max_frames));

        Self {
            input_buffer,
            output_buffer,
            channel_layout: Some(channel_layout),
            num_frames,
        }
    }

    pub fn for_each_channel(&mut self, mut f: impl FnMut(&[T], &mut [T])) {
        match &self.channel_layout {
            Some(layout) => {
                for ch in layout.iter() {
                    f(
                        self.input_buffer.channel(ch).unwrap(),
                        self.output_buffer.channel_mut(ch).unwrap(),
                    );
                }
            }
            None => {
                let num_channels = self
                    .input_buffer
                    .num_channels()
                    .min(self.output_buffer.num_channels());

                for ch in 0..num_channels {
                    f(
                        self.input_buffer.channel(ch).unwrap(),
                        self.output_buffer.channel_mut(ch).unwrap(),
                    );
                }
            }
        }
    }
}

pub struct PassThrough;

impl<T: Sample> Processor<T> for PassThrough {
    fn process(&mut self, context: &mut ProcessingContext<T>) {
        context.for_each_channel(|input, output| {
            output.copy_from_slice(input);
        });
    }
}

pub struct NoOp;

impl<T: Sample> Processor<T> for NoOp {
    fn process(&mut self, _context: &mut ProcessingContext<T>) {}
}
