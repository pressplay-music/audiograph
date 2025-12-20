use crate::{channel::ChannelLayout, sample::Sample};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize(pub usize);

// TODO: add more creation methods
// TODO: clear() return type
pub trait AudioBuffer<T: Sample> {
    fn num_channels(&self) -> usize;
    fn num_frames(&self) -> FrameSize;
    fn channel(&self, index: usize) -> Option<&[T]>;
    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]>;

    fn clear(&mut self);

    fn copy_to_interleaved(&self, output: &mut [T]) {
        for channel in 0..self.num_channels() {
            let src_channel = self.channel(channel).unwrap();
            for (frame, &sample) in src_channel.iter().enumerate() {
                output[frame * self.num_channels() + channel] = sample;
            }
        }
    }

    fn add(&mut self, other: &dyn AudioBuffer<T>, channel_layout: &Option<ChannelLayout>) {
        if let Some(layout) = channel_layout {
            // TODO: check if take() is used correctly here
            for channel in layout.iter().take(self.num_channels()) {
                if let (Some(src), Some(dst)) = (other.channel(channel), self.channel_mut(channel))
                {
                    dst.iter_mut().zip(src.iter()).for_each(|(a, b)| {
                        *a += *b;
                    });
                }
            }
        } else {
            let num_channels = self.num_channels().min(other.num_channels());
            for channel in 0..num_channels {
                let src = other.channel(channel).unwrap();
                let dst = self.channel_mut(channel).unwrap();
                dst.iter_mut().zip(src.iter()).for_each(|(a, b)| {
                    *a += *b;
                });
            }
        }
    }
}

pub struct MultiChannelBuffer<T: Sample> {
    channels: Vec<Box<[T]>>,
    num_frames: FrameSize,
}

impl<T: Sample> MultiChannelBuffer<T> {
    pub fn new(num_channels: usize, num_frames: FrameSize) -> Self {
        let mut channels = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            channels.push(vec![T::zero(); num_frames.0].into_boxed_slice());
        }
        Self {
            channels,
            num_frames,
        }
    }
}

impl<T: Sample> AudioBuffer<T> for MultiChannelBuffer<T> {
    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    fn num_frames(&self) -> FrameSize {
        self.num_frames
    }

    fn channel(&self, index: usize) -> Option<&[T]> {
        self.channels.get(index).map(|b| &**b)
    }

    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.channels.get_mut(index).map(|b| &mut **b)
    }

    fn clear(&mut self) {
        for channel in self.channels.iter_mut() {
            for sample in channel.iter_mut() {
                *sample = T::zero();
            }
        }
    }
}

/// Non-owning view into a channel-based collection of audio samples.
///
/// Useful for zero-copy processing of immutable (input) audio data. Example:
/// fn channel_based_callback<'a>(data: &[&[f32]]) {
///     let buffer_view = MultiChannelBufferView::new(data, FrameSize(data[0].len()));
/// }
pub struct MultiChannelBufferView<'a, T: Sample> {
    channels: &'a [&'a [T]],
    num_frames: FrameSize,
}

impl<'a, T: Sample> MultiChannelBufferView<'a, T> {
    pub fn new(channels: &'a [&'a [T]], num_frames: FrameSize) -> Self {
        Self {
            channels,
            num_frames,
        }
    }
}

impl<T: Sample> AudioBuffer<T> for MultiChannelBufferView<'_, T> {
    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    fn num_frames(&self) -> FrameSize {
        self.num_frames
    }

    fn channel(&self, index: usize) -> Option<&[T]> {
        self.channels.get(index).map(|b| &**b)
    }

    fn channel_mut(&mut self, _index: usize) -> Option<&mut [T]> {
        None
    }

    fn clear(&mut self) {}
}

/// Non-owning mutable view into a channel-based collection of audio samples.
///
/// Useful for zero-copy processing of mutable (output) audio data. Example:
/// fn channel_based_callback<'a>(data: &'a mut [&'a mut [f32]]) {
///     let mut mutable_buffer_view = MultiChannelBufferViewMut::new(data, FrameSize(data[0].len()));
/// }
pub struct MultiChannelBufferViewMut<'a, T: Sample> {
    channels: &'a mut [&'a mut [T]],
    num_frames: FrameSize,
}

impl<'a, T: Sample> MultiChannelBufferViewMut<'a, T> {
    pub fn new(channels: &'a mut [&'a mut [T]], num_frames: FrameSize) -> Self {
        assert!(!channels.is_empty());
        Self {
            channels,
            num_frames,
        }
    }
}

impl<T: Sample> AudioBuffer<T> for MultiChannelBufferViewMut<'_, T> {
    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    fn num_frames(&self) -> FrameSize {
        self.num_frames
    }

    fn channel(&self, index: usize) -> Option<&[T]> {
        self.channels.get(index).map(|b| &**b)
    }

    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.channels.get_mut(index).map(|b| &mut **b)
    }

    fn clear(&mut self) {
        for channel in &mut *self.channels {
            for sample in channel.iter_mut() {
                *sample = T::zero();
            }
        }
    }
}

// Immutable AudioBuffer view that remaps channel indices
pub struct RewiredBufferView<'a, T: Sample> {
    pub buffer: &'a dyn AudioBuffer<T>,
    pub rewire: &'a HashMap<usize, usize>,
}

impl<T: Sample> AudioBuffer<T> for RewiredBufferView<'_, T> {
    fn num_channels(&self) -> usize {
        self.rewire.keys().max().map_or(0, |&max| max + 1)
    }

    fn num_frames(&self) -> FrameSize {
        self.buffer.num_frames()
    }

    fn channel(&self, index: usize) -> Option<&[T]> {
        if let Some(&source_channel) = self.rewire.get(&index) {
            self.buffer.channel(source_channel)
        } else {
            None
        }
    }

    fn channel_mut(&mut self, _index: usize) -> Option<&mut [T]> {
        None
    }

    fn clear(&mut self) {
        panic!("Cannot clear an immutable buffer view");
    }
}
