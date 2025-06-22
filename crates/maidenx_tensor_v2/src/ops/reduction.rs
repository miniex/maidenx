use crate::Tensor;
use maidenx_core::scalar::Scalar;

impl Tensor {
    pub fn sum(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Tensor {
        self.try_sum(dim, keep_dim).expect("failed to sum tensor")
    }

    pub fn sum_all(&self) -> Self {
        self.try_sum_all().expect("failed to sum all tensor")
    }

    pub fn sum_to_shape(&self, shape: &[usize]) -> Tensor {
        self.try_sum_to_shape(shape).expect("failed to sum to shape")
    }

    pub fn mean(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Self {
        self.try_mean(dim, keep_dim).expect("failed to mean tensor")
    }

    pub fn mean_all(&self) -> Self {
        self.try_mean_all().expect("failed to mean all tensor")
    }

    pub fn fold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Tensor {
        self.try_fold(dim, size, step).expect("failed to fold tensor")
    }

    pub fn max(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Tensor {
        self.try_max(dim, keep_dim).expect("failed to max tensor")
    }

    pub fn max_all(&self) -> Self {
        self.try_max_all().expect("failed to max all tensor")
    }

    pub fn min(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Tensor {
        self.try_min(dim, keep_dim).expect("failed to min tensor")
    }

    pub fn min_all(&self) -> Self {
        self.try_min_all().expect("failed to min all tensor")
    }

    pub fn norm(&self, p: impl Into<Scalar>, dim: impl Into<Scalar>, keep_dim: bool) -> Self {
        self.try_norm(p, dim, keep_dim).expect("failed to norm tensor")
    }

    pub fn norm_all(&self, p: impl Into<Scalar>) -> Tensor {
        self.try_norm_all(p).expect("failed to norm all tensor")
    }

    pub fn var(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Self {
        self.try_var(dim, keep_dim, unbiased).expect("failed to var tensor")
    }

    pub fn var_all(&self, unbiased: bool) -> Tensor {
        self.try_var_all(unbiased).expect("failed to var all tensor")
    }

    pub fn std(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Tensor {
        self.try_std(dim, keep_dim, unbiased).expect("failed to std tensor")
    }

    pub fn std_all(&self, unbiased: bool) -> Tensor {
        self.try_std_all(unbiased).expect("failed to std all tensor")
    }
}
