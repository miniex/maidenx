#[cfg(feature = "nn")]
pub mod nn;
pub mod ops;

type CleanupFn = Option<Box<dyn FnOnce()>>;
