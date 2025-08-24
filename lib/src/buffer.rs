use crate::{channel::ChannelLayout, sample::Sample};

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

    fn add(&mut self, other: &dyn AudioBuffer<T>, channel_layout: ChannelLayout) {
        let max_num_channels = self.num_channels().min(other.num_channels());
        for channel in channel_layout.iter() {
            if channel >= max_num_channels {
                break;
            }
            self.channel_mut(channel)
                .unwrap()
                .iter_mut()
                .zip(other.channel(channel).unwrap().iter())
                .for_each(|(a, b)| {
                    *a += *b;
                });
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

pub struct MultiChannelBufferView<'a, T: Sample> {
    channels: &'a [Box<[T]>],
    num_frames: FrameSize,
}

impl<'a, T: Sample> MultiChannelBufferView<'a, T> {
    pub fn new(channels: &'a [Box<[T]>], num_frames: FrameSize) -> Self {
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
