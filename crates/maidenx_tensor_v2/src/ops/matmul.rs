use crate::Tensor;

impl Tensor {
    pub fn matmul(&self, rhs: &Self) -> Tensor {
        self.try_matmul(rhs).expect("failed to perform matrix multiplication")
    }
}
