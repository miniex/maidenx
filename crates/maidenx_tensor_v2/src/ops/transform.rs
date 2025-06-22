use crate::Tensor;
use maidenx_core::scalar::Scalar;

impl Tensor {
    pub fn view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Self {
        self.try_view(shape).expect("failed to create view")
    }

    pub fn squeeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Self {
        self.try_squeeze(dims).expect("failed to squeeze tensor")
    }

    pub fn squeeze_all(&self) -> Self {
        self.try_squeeze_all().expect("failed to squeeze all dimensions")
    }

    pub fn unsqueeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Self {
        self.try_unsqueeze(dims).expect("failed to unsqueeze tensor")
    }

    pub fn transpose(&self, dim0: impl Into<Scalar>, dim1: impl Into<Scalar>) -> Self {
        self.try_transpose(dim0, dim1).expect("failed to transpose tensor")
    }

    pub fn slice(
        &self,
        dim: impl Into<Scalar>,
        start: impl Into<Scalar>,
        end: Option<impl Into<Scalar>>,
        step: impl Into<Scalar>,
    ) -> Self {
        self.try_slice(dim, start, end, step).expect("failed to slice tensor")
    }

    pub fn unfold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Self {
        self.try_unfold(dim, size, step).expect("failed to unfold tensor")
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        self.try_broadcast(shape).expect("failed to broadcast tensor")
    }

    pub fn broadcast_like(&self, other: &Self) -> Self {
        self.try_broadcast_like(other)
            .expect("failed to broadcast tensor like other")
    }

    pub fn broadcast_left(&self, batch_dims: &[usize]) -> Self {
        self.try_broadcast_left(batch_dims)
            .expect("failed to broadcast tensor left")
    }

    pub fn reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Self {
        self.try_reshape(shape).expect("failed to reshape tensor")
    }
}
