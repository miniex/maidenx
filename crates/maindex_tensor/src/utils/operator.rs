use crate::Tensor;
use std::ops::{Add, Div, Mul, Sub};

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        self._ops_add(&other).unwrap()
    }
}
impl<'a> Add<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: &'a Tensor) -> Self::Output {
        self._ops_add(other).unwrap()
    }
}
impl<'a> Add<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        self._ops_add(&other).unwrap()
    }
}
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Self::Output {
        self._ops_add(other).unwrap()
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        self._ops_sub(&other).unwrap()
    }
}
impl<'a> Sub<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &'a Tensor) -> Self::Output {
        self._ops_sub(other).unwrap()
    }
}
impl<'a> Sub<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        self._ops_sub(&other).unwrap()
    }
}
impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: &'b Tensor) -> Self::Output {
        self._ops_sub(other).unwrap()
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        self._ops_mul(&other).unwrap()
    }
}
impl<'a> Mul<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: &'a Tensor) -> Self::Output {
        self._ops_mul(other).unwrap()
    }
}
impl<'a> Mul<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        self._ops_mul(&other).unwrap()
    }
}
impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: &'b Tensor) -> Self::Output {
        self._ops_mul(other).unwrap()
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        self._ops_div(&other).unwrap()
    }
}
impl<'a> Div<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: &'a Tensor) -> Self::Output {
        self._ops_div(other).unwrap()
    }
}
impl<'a> Div<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        self._ops_div(&other).unwrap()
    }
}
impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, other: &'b Tensor) -> Self::Output {
        self._ops_div(other).unwrap()
    }
}
