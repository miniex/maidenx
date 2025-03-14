use crate::Tensor;
use std::ops::{Add, Div, Mul, Sub};

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs).unwrap()
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(self, &rhs).unwrap()
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(&self, rhs).unwrap()
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(&self, &rhs).unwrap()
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs).unwrap()
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(self, &rhs).unwrap()
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(&self, rhs).unwrap()
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(&self, &rhs).unwrap()
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs).unwrap()
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::mul(self, &rhs).unwrap()
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(&self, rhs).unwrap()
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::mul(&self, &rhs).unwrap()
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(self, rhs).unwrap()
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::div(self, &rhs).unwrap()
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(&self, rhs).unwrap()
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::div(&self, &rhs).unwrap()
    }
}
