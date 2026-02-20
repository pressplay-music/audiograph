use crate::{
    buffer::{AudioBuffer, FrameSize},
    channel::ChannelSelection,
    sample::Sample,
};

/// Trait for audio processing nodes
///
/// A processor implements the `process` method that operates directly on audio data
/// provided through the `ProcessingContext`.
pub trait Processor<T: Sample> {
    /// Processes audio data from the input buffer to the output buffer of the provided context
    fn process(&mut self, context: &mut ProcessingContext<T>);
}

/// The audio processing context used by a [`Processor`] to perform its processing operation
///
/// The context includes references to input and output buffers, an optional channel selection
/// to specify which channels to process, and the number of frames to process.
#[non_exhaustive]
pub struct ProcessingContext<'a, T: Sample> {
    pub input_buffer: &'a dyn AudioBuffer<T>,
    pub output_buffer: &'a mut dyn AudioBuffer<T>,
    pub channel_selection: Option<ChannelSelection>,

    /// Number of frames to process from each channel. This may be less than or equal to the total number of frames
    /// available in the buffers.
    pub num_frames: FrameSize,
}

impl<'a, T: Sample> ProcessingContext<'a, T> {
    /// Creates a new processing context without validation
    ///
    /// This constructor does not validate that the channel selection and frame size
    /// are within the bounds of the provided buffers. Use this method when you have
    /// already validated these parameters.
    pub fn create_unchecked(
        input_buffer: &'a dyn AudioBuffer<T>,
        output_buffer: &'a mut dyn AudioBuffer<T>,
        channel_selection: Option<ChannelSelection>,
        num_frames: FrameSize,
    ) -> Self {
        Self {
            input_buffer,
            output_buffer,
            channel_selection,
            num_frames,
        }
    }

    /// Creates a new processing context with validation and clamping of channel selection and frame size
    ///
    /// This constructor automatically adjusts the channel selection and frame count to ensure they do not exceed
    /// the capabilities of the provided buffers.
    pub fn create_checked(
        input_buffer: &'a dyn AudioBuffer<T>,
        output_buffer: &'a mut dyn AudioBuffer<T>,
        mut channel_selection: ChannelSelection,
        mut num_frames: FrameSize,
    ) -> Self {
        let max_channels = input_buffer
            .num_channels()
            .min(output_buffer.num_channels());
        channel_selection.clamp(max_channels);

        let max_frames = input_buffer
            .num_frames()
            .0
            .min(output_buffer.num_frames().0);
        num_frames = FrameSize(num_frames.0.min(max_frames));

        Self {
            input_buffer,
            output_buffer,
            channel_selection: Some(channel_selection),
            num_frames,
        }
    }

    /// Iterates over selected channels, applying a function to each input/output channel pair
    ///
    /// This helper method simplifies the common pattern of processing audio data channel by channel.
    /// If a channel selection is specified, only those channels are processed. Otherwise, all channels
    /// up to the minimum of input and output channel counts are processed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// context.for_each_channel(|input, output| {
    ///     for (i, o) in input.iter().zip(output.iter_mut()) {
    ///         *o = *i * 0.5; // Apply 0.5 gain
    ///     }
    /// });
    /// ```
    pub fn for_each_channel(&mut self, mut f: impl FnMut(&[T], &mut [T])) {
        match &self.channel_selection {
            Some(selection) => {
                for ch in selection.iter() {
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

/// A passthrough processor that copies input to output without modification
pub struct PassThrough;

impl<T: Sample> Processor<T> for PassThrough {
    fn process(&mut self, context: &mut ProcessingContext<T>) {
        context.for_each_channel(|input, output| {
            output.copy_from_slice(input);
        });
    }
}

/// A no-operation processor that does nothing
pub struct NoOp;

impl<T: Sample> Processor<T> for NoOp {
    fn process(&mut self, _context: &mut ProcessingContext<T>) {}
}
