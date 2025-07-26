pub trait Sample: num::Float + Default + std::ops::Add + std::ops::AddAssign {}
impl<T> Sample for T where T: num::Float + Default + std::ops::Add + std::ops::AddAssign {}
