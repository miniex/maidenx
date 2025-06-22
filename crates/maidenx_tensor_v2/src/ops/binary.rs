use crate::Tensor;

impl Tensor {
    pub fn add(&self, rhs: &Self) -> Self {
        self.try_add(rhs).expect("failed to add tensors")
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.try_sub(rhs).expect("failed to subtract tensors")
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.try_mul(rhs).expect("failed to multiply tensors")
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.try_div(rhs).expect("failed to divide tensors")
    }

    pub fn maximum(&self, rhs: &Self) -> Self {
        self.try_maximum(rhs).expect("failed to compute maximum")
    }

    pub fn minimum(&self, rhs: &Self) -> Self {
        self.try_minimum(rhs).expect("failed to compute minimum")
    }

    pub fn logical_and(&self, rhs: &Self) -> Self {
        self.try_logical_and(rhs).expect("failed to compute logical_and")
    }

    pub fn logical_or(&self, rhs: &Self) -> Self {
        self.try_logical_or(rhs).expect("failed to compute logical_or")
    }

    pub fn logical_xor(&self, rhs: &Self) -> Self {
        self.try_logical_xor(rhs).expect("failed to compute logical_xor")
    }

    pub fn eq(&self, rhs: &Self) -> Self {
        self.try_eq(rhs).expect("failed to compute eq")
    }

    pub fn ne(&self, rhs: &Self) -> Self {
        self.try_ne(rhs).expect("failed to compute ne")
    }

    pub fn lt(&self, rhs: &Self) -> Self {
        self.try_lt(rhs).expect("failed to compute lt")
    }

    pub fn le(&self, rhs: &Self) -> Self {
        self.try_le(rhs).expect("failed to compute le")
    }

    pub fn gt(&self, rhs: &Self) -> Self {
        self.try_gt(rhs).expect("failed to compute gt")
    }

    pub fn ge(&self, rhs: &Self) -> Self {
        self.try_ge(rhs).expect("failed to compute ge")
    }

    pub fn add_(&mut self, rhs: &Self) {
        self.try_add_(rhs).expect("failed to add_ tensors")
    }

    pub fn sub_(&mut self, rhs: &Self) {
        self.try_sub_(rhs).expect("failed to sub_ tensors")
    }

    pub fn mul_(&mut self, rhs: &Self) {
        self.try_mul_(rhs).expect("failed to mul_ tensors")
    }

    pub fn div_(&mut self, rhs: &Self) {
        self.try_div_(rhs).expect("failed to div_ tensors")
    }
}
