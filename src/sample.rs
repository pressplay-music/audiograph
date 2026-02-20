/// A floating-point sample type that can be used for audio processing
pub trait Sample: num::Float + Default + std::ops::Add + std::ops::AddAssign {}
impl<T> Sample for T where T: num::Float + Default + std::ops::Add + std::ops::AddAssign {}
