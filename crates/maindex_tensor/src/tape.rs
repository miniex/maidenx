use crate::Tensor;
use std::cell::RefCell;

thread_local! {
    pub static TENSOR_TAPE: RefCell<TensorTape> = RefCell::new(TensorTape::new());
}

#[derive(Default)]
pub struct TensorTape {
    tensors: Vec<Tensor>,
}

impl TensorTape {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    pub fn add(&mut self, tensor: Tensor) -> Tensor {
        self.tensors.push(tensor.clone());
        tensor
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.tensors.clear();
    }
}
