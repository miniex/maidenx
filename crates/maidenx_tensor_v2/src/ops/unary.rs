use crate::Tensor;
use maidenx_core::scalar::Scalar;

impl Tensor {
    pub fn neg(&self) -> Self {
        self.try_neg().expect("failed to negate tensor")
    }

    pub fn abs(&self) -> Self {
        self.try_abs().expect("failed to compute absolute value")
    }

    pub fn sign(&self) -> Self {
        self.try_sign().expect("failed to compute sign")
    }

    pub fn square(&self) -> Self {
        self.try_square().expect("failed to compute square")
    }

    pub fn sqrt(&self) -> Self {
        self.try_sqrt().expect("failed to compute sqrt")
    }

    pub fn relu(&self) -> Self {
        self.try_relu().expect("failed to compute relu")
    }

    pub fn sigmoid(&self) -> Self {
        self.try_sigmoid().expect("failed to compute sigmoid")
    }

    pub fn tanh(&self) -> Self {
        self.try_tanh().expect("failed to compute tanh")
    }

    pub fn gelu(&self) -> Self {
        self.try_gelu().expect("failed to compute gelu")
    }

    pub fn sin(&self) -> Self {
        self.try_sin().expect("failed to compute sin")
    }

    pub fn cos(&self) -> Self {
        self.try_cos().expect("failed to compute cos")
    }

    pub fn tan(&self) -> Self {
        self.try_tan().expect("failed to compute tan")
    }

    pub fn ln(&self) -> Self {
        self.try_ln().expect("failed to compute ln")
    }

    pub fn log(&self) -> Self {
        self.try_log().expect("failed to compute log")
    }

    pub fn log10(&self) -> Self {
        self.try_log10().expect("failed to compute log10")
    }

    pub fn log2(&self) -> Self {
        self.try_log2().expect("failed to compute log2")
    }

    pub fn exp(&self) -> Self {
        self.try_exp().expect("failed to compute exp")
    }

    pub fn exp10(&self) -> Self {
        self.try_exp10().expect("failed to compute exp10")
    }

    pub fn exp2(&self) -> Self {
        self.try_exp2().expect("failed to compute exp2")
    }

    pub fn softplus(&self) -> Self {
        self.try_softplus().expect("failed to compute softplus")
    }

    pub fn recip(&self) -> Self {
        self.try_recip().expect("failed to compute recip")
    }

    pub fn logical_not(&self) -> Self {
        self.try_logical_not().expect("failed to compute logical_not")
    }

    pub fn add_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_add_scalar(scalar).expect("failed to add scalar")
    }

    pub fn sub_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_sub_scalar(scalar).expect("failed to subtract scalar")
    }

    pub fn mul_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_mul_scalar(scalar).expect("failed to multiply by scalar")
    }

    pub fn div_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_div_scalar(scalar).expect("failed to divide by scalar")
    }

    pub fn maximum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_maximum_scalar(scalar)
            .expect("failed to compute maximum_scalar")
    }

    pub fn minimum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_minimum_scalar(scalar)
            .expect("failed to compute minimum_scalar")
    }

    pub fn pow<T: Into<Scalar>>(&self, exponent: T) -> Self {
        self.try_pow(exponent).expect("failed to compute pow")
    }

    pub fn leaky_relu<T: Into<Scalar>>(&self, alpha: T) -> Self {
        self.try_leaky_relu(alpha).expect("failed to compute leaky_relu")
    }

    pub fn elu<T: Into<Scalar>>(&self, alpha: T) -> Self {
        self.try_elu(alpha).expect("failed to compute elu")
    }

    pub fn eq_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_eq_scalar(scalar).expect("failed to compute eq_scalar")
    }

    pub fn ne_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_ne_scalar(scalar).expect("failed to compute ne_scalar")
    }

    pub fn lt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_lt_scalar(scalar).expect("failed to compute lt_scalar")
    }

    pub fn le_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_le_scalar(scalar).expect("failed to compute le_scalar")
    }

    pub fn gt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_gt_scalar(scalar).expect("failed to compute gt_scalar")
    }

    pub fn ge_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_ge_scalar(scalar).expect("failed to compute ge_scalar")
    }
}
