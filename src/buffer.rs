use crate::{channel::ChannelLayout, sample::Sample};

// TODO: add channel iterators and more creation methods
// TODO: clear() return type
pub trait AudioBuffer<T: Sample> {
    fn num_channels(&self) -> usize;
    fn num_frames(&self) -> usize;
    fn channel(&self, index: usize) -> Option<&[T]>;
    fn channel_mut(&mut self, index: usize) -> Option<&mut [T]>;

    fn clear(&mut self);
    fn add(&mut self, other: &dyn AudioBuffer<T>, _channel_layout: ChannelLayout) {
        for channel in 0..self.num_channels() {
            if let (Some(self_channel), Some(other_channel)) =
                (self.channel_mut(channel), other.channel(channel))
            {
                self_channel
                    .iter_mut()
                    .zip(other_channel.iter())
                    .for_each(|(a, b)| {
                        *a += *b;
                    });
            }
        }
    }
}

pub struct ChannelIter<'a, T: Sample> {
    buffer: &'a dyn AudioBuffer<T>,
    channel_index: usize,
}

impl<'a, T: Sample> ChannelIter<'a, T> {
    pub fn new(buffer: &'a dyn AudioBuffer<T>, channel_index: usize) -> Self {
        Self {
            buffer,
            channel_index,
        }
    }
}

impl<'a, T: Sample> Iterator for ChannelIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.channel_index < self.buffer.num_channels() {
            let channel = self.buffer.channel(self.channel_index);
            self.channel_index += 1;
            channel
        } else {
            None
        }
    }
}

pub fn iter_channels<'a, T: Sample>(buffer: &'a dyn AudioBuffer<T>) -> ChannelIter<'a, T> {
    ChannelIter::new(buffer, 0)
}
pub struct ChannelIterMut<'a, T: Sample> {
    buffer: &'a mut dyn AudioBuffer<T>,
    channel_index: usize,
}

impl<'a, T: Sample> ChannelIterMut<'a, T> {
    pub fn new(buffer: &'a mut dyn AudioBuffer<T>, channel_index: usize) -> Self {
        Self {
            buffer,
            channel_index,
        }
    }
}

impl<'a, T: Sample> Iterator for ChannelIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.channel_index < self.buffer.num_channels() {
            let channel = self.buffer.channel_mut(self.channel_index);
            self.channel_index += 1;
            channel
        } else {
            None
        }
    }
}

pub fn iter_channels_mut<'a, T: Sample>(
    buffer: &'a mut dyn AudioBuffer<T>,
) -> ChannelIterMut<'a, T> {
    ChannelIterMut::new(buffer, 0)
}

pub struct MultiChannelBuffer<T: Sample> {
    channels: Vec<Box<[T]>>,
    num_frames: usize,
}

impl<T: Sample> MultiChannelBuffer<T> {
    pub fn new(num_channels: usize, num_frames: usize) -> Self {
        let mut channels = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            channels.push(vec![T::zero(); num_frames].into_boxed_slice());
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

    fn num_frames(&self) -> usize {
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
    num_frames: usize,
}

impl<T: Sample> AudioBuffer<T> for MultiChannelBufferView<'_, T> {
    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    fn num_frames(&self) -> usize {
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
