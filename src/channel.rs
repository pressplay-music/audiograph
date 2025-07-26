#[derive(Clone)]
pub struct ChannelLayout {}

impl ChannelLayout {
    pub fn compatible(&self, _other: &ChannelLayout) -> bool {
        // TODO: implement meaningful check
        true
    }
}
