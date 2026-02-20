use bitvec::{array::BitArray, slice::IterOnes};

include!(concat!(env!("OUT_DIR"), "/constants.rs"));

const WORDS: usize = (MAX_CHANNELS - 1) / 64 + 1;
pub type ChannelSelection = ChannelSelectionImpl<MAX_CHANNELS, WORDS>;

#[derive(Clone)]
pub struct ChannelSelectionImpl<const NUM_CHANNELS: usize, const WORDS: usize> {
    bits: BitArray<[u64; WORDS]>,
}

impl<const NUM_CHANNELS: usize, const WORDS: usize> Default
    for ChannelSelectionImpl<NUM_CHANNELS, WORDS>
{
    fn default() -> Self {
        Self::new(NUM_CHANNELS)
    }
}

impl<const NUM_CHANNELS: usize, const WORDS: usize> ChannelSelectionImpl<NUM_CHANNELS, WORDS> {
    pub fn new(num_connected: usize) -> Self {
        let mut bits = BitArray::ZERO;
        let count = num_connected.min(NUM_CHANNELS);
        bits[..count].fill(true);
        Self { bits }
    }

    pub fn from_indices(indices: &[usize]) -> Self {
        let mut selection = ChannelSelectionImpl::new(0);
        for &index in indices {
            selection.connect(index);
        }
        selection
    }

    // TODO: return Result
    pub fn connect(&mut self, channel: usize) {
        if channel < NUM_CHANNELS {
            self.bits.set(channel, true);
        }
    }

    pub fn is_connected(&self, channel: usize) -> bool {
        if channel < NUM_CHANNELS {
            self.bits[channel]
        } else {
            false
        }
    }

    pub fn iter(&self) -> IterOnes<'_, u64, bitvec::order::Lsb0> {
        self.bits.iter_ones()
    }

    pub fn clamp(&mut self, max_channels: usize) {
        self.bits.split_at_mut(max_channels).1.fill(false);
    }

    pub fn index_of_last_connected(&self) -> Option<usize> {
        self.bits.last_one()
    }

    pub fn combine(&mut self, other: &Self) {
        self.bits |= other.bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_selection_creation() {
        let selection = ChannelSelection::new(3);
        assert!(selection.is_connected(0));
        assert!(selection.is_connected(1));
        assert!(selection.is_connected(2));
        assert!(!selection.is_connected(3));
        assert!(!selection.is_connected(63));
        assert!(!selection.is_connected(64));
    }

    #[test]
    fn test_channel_selection_clamping() {
        let mut selection = ChannelSelection::new(5);
        selection.clamp(3);
        assert!(selection.is_connected(0));
        assert!(selection.is_connected(1));
        assert!(selection.is_connected(2));
        assert!(!selection.is_connected(3));
        assert!(!selection.is_connected(4));
    }
}
