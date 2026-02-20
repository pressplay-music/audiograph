use bitvec::{array::BitArray, slice::IterOnes};

include!(concat!(env!("OUT_DIR"), "/constants.rs"));

const WORDS: usize = (MAX_CHANNELS - 1) / 64 + 1;

/// Compact representation of an audio channel selection
///
/// Selected channel indices are encoded using a bit array, leading to a very small memory footprint.
/// A channel selection is meant to be copied wherever it is needed. It includes methods for checking
/// and modifying a channel selection status, as well as iterating over selected channel indices.
///
/// In the context of audio processing, a channel selection is synonymous with a "channel connection" between
/// input and output buffers.
pub type ChannelSelection = ChannelSelectionImpl<MAX_CHANNELS, WORDS>;

#[derive(Clone)]
pub struct ChannelSelectionImpl<const NUM_CHANNELS: usize, const WORDS: usize> {
    bits: BitArray<[u64; WORDS]>,
}

impl<const NUM_CHANNELS: usize, const WORDS: usize> Default
    for ChannelSelectionImpl<NUM_CHANNELS, WORDS>
{
    /// Creates a default selection, where every channel is selected
    fn default() -> Self {
        Self::new(NUM_CHANNELS)
    }
}

// TODO: disconnect()
impl<const NUM_CHANNELS: usize, const WORDS: usize> ChannelSelectionImpl<NUM_CHANNELS, WORDS> {
    /// Creates a channel selection with the first `num_connected` channels connected and remaining channels disconnected
    pub fn new(num_connected: usize) -> Self {
        let mut bits = BitArray::ZERO;
        let count = num_connected.min(NUM_CHANNELS);
        bits[..count].fill(true);
        Self { bits }
    }

    /// Creates a channel selection from a slice of connected channel indices
    ///
    /// # Example
    /// ```rust,ignore
    /// let selection = ChannelSelection::from_indices(&[0, 2, 4]);
    /// ```
    pub fn from_indices(indices: &[usize]) -> Self {
        let mut selection = ChannelSelectionImpl::new(0);
        for &index in indices {
            selection.connect(index);
        }
        selection
    }

    /// Defines a specific channel as selected / connected
    // TODO: return Result
    pub fn connect(&mut self, channel: usize) {
        if channel < NUM_CHANNELS {
            self.bits.set(channel, true);
        }
    }

    /// Checks if a specifc channel is selected / connected
    pub fn is_connected(&self, channel: usize) -> bool {
        if channel < NUM_CHANNELS {
            self.bits[channel]
        } else {
            false
        }
    }

    /// Create iterator over all selected / connected channels
    pub fn iter(&self) -> IterOnes<'_, u64, bitvec::order::Lsb0> {
        self.bits.iter_ones()
    }

    /// Unselects / disconnects all channels at index `max_channels` and above
    pub fn clamp(&mut self, max_channels: usize) {
        self.bits.split_at_mut(max_channels).1.fill(false);
    }

    /// Returns the highest index of all currently selected / connected channels
    pub fn index_of_last_connected(&self) -> Option<usize> {
        self.bits.last_one()
    }

    /// Combines this channel selection with `other` (bitwise OR operation)
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
