use crate::Tensor;
use maidenx_core::scalar::Scalar;

impl Tensor {
    pub fn pad(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Tensor {
        self.try_pad(paddings, pad_value).expect("failed to pad tensor")
    }

    pub fn pad_with_constant(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Tensor {
        self.try_pad_with_constant(paddings, pad_value)
            .expect("failed to pad tensor with constant")
    }

    pub fn pad_with_reflection(&self, paddings: &[(usize, usize)]) -> Tensor {
        self.try_pad_with_reflection(paddings)
            .expect("failed to pad tensor with reflection")
    }

    pub fn pad_with_replication(&self, paddings: &[(usize, usize)]) -> Tensor {
        self.try_pad_with_replication(paddings)
            .expect("failed to pad tensor with replication")
    }
}
