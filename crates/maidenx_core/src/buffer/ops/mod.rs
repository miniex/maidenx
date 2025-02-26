pub mod binary;
pub mod matmul;
pub mod reduction;
pub mod unary;

type CleanupFn = Option<Box<dyn FnOnce()>>;
