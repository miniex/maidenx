use crate::{
    utils::promotion::{promote_scalar_for_tensor, promote_tensor},
    Tensor, TensorNode,
};
use maidenx_core::{dtype::DType, error::Result, scalar::Scalar};

impl Tensor {
    pub fn neg(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };
        let target_dtype = target_dtype.to_signed();
        let input = self.to_dtype(target_dtype)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::neg(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.neg()?])
            });
            let node = TensorNode::new("neg".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn abs(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };
        let input = self.to_dtype(target_dtype)?;

        let mut result = if target_dtype.is_uint() {
            input.clone()
        } else {
            let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;
            unsafe {
                result.with_buffer_mut(|out_buf| {
                    maidenx_core::be::ops::unary::abs(
                        out_buf,
                        input.buffer(),
                        input.size(),
                        input.ndim(),
                        Some(&prepare_metadata(&input)),
                    )?;
                    Ok(())
                })?;
            }
            result
        };

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&input.sign()?)?])
            });
            let node = TensorNode::new("abs".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sign(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };
        let input = self.to_dtype(target_dtype)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::sign(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn square(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };
        let input = self.to_dtype(target_dtype)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::square(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul_scalar(2.0)?.mul(&input)?])
            });
            let node = TensorNode::new("square".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };
        let input = promote_tensor(self, target_dtype)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::sqrt(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.div(&output.mul_scalar(2.0)?)?])
            });
            let node = TensorNode::new("sqrt".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn relu(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };
        let input = promote_tensor(self, target_dtype)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::relu(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&input.gt_scalar(0.0)?.to_dtype(output.dtype())?)?])
            });
            let node = TensorNode::new("relu".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sigmoid(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::sigmoid(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&output)?.mul(&output.sub_scalar(1.0)?.neg()?)?])
            });
            let node = TensorNode::new("sigmoid".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn tanh(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::tanh(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&output.pow(2.0)?.neg()?.add_scalar(1.0)?)?])
            });
            let node = TensorNode::new("tanh".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn gelu(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::gelu(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let sqrt_2_over_pi = 0.7978845608028654;
                let coeff = 0.044715;

                let x_squared = input.mul(&input)?;
                let x_cubed = x_squared.mul(&input)?;
                let tanh_arg = input.add(&x_cubed.mul_scalar(coeff)?)?.mul_scalar(sqrt_2_over_pi)?;

                let tanh_val = tanh_arg.tanh()?;
                let sech_squared = tanh_val.mul(&tanh_val)?.mul_scalar(-1.0)?.add_scalar(1.0)?;
                let inner_derivative = x_squared
                    .mul_scalar(3.0 * coeff)?
                    .add_scalar(1.0)?
                    .mul_scalar(sqrt_2_over_pi)?;

                let term1 = tanh_val.add_scalar(1.0)?.mul_scalar(0.5)?;
                let term2 = input.mul(&sech_squared)?.mul(&inner_derivative)?.mul_scalar(0.5)?;
                let grad = grad_out.mul(&term1.add(&term2)?)?;

                Ok(vec![grad])
            });
            let node = TensorNode::new("gelu".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sin(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::sin(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&input.cos()?)?])
            });
            let node = TensorNode::new("sin".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn cos(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::cos(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&input.sin()?.neg()?)?])
            });
            let node = TensorNode::new("cos".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn tan(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::tan(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let cos_x = input.cos()?;
                let sec_squared = cos_x.mul(&cos_x)?.pow(-1.0)?;
                Ok(vec![grad_out.mul(&sec_squared)?])
            });
            let node = TensorNode::new("tan".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn ln(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::ln(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.div(&input)?])
            });
            let node = TensorNode::new("ln".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn log(&self) -> Result<Tensor> {
        self.ln()
    }

    pub fn log10(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::log10(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let ln10 = Scalar::from(std::f32::consts::LN_10); // ln(10)
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.div(&input)?.div_scalar(ln10)?])
            });
            let node = TensorNode::new("log10".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn log2(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::log2(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let ln2 = Scalar::from(std::f32::consts::LN_2); // ln(2)
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.div(&input)?.div_scalar(ln2)?])
            });
            let node = TensorNode::new("log2".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn exp(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::exp(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&output)?])
            });
            let node = TensorNode::new("exp".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn exp10(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::exp10(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();
            let ln10 = Scalar::from(std::f32::consts::LN_10); // ln(10)
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&output)?.mul_scalar(ln10)?])
            });
            let node = TensorNode::new("exp10".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn exp2(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::exp2(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();
            let ln2 = Scalar::from(std::f32::consts::LN_2); // ln(2)
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&output)?.mul_scalar(ln2)?])
            });
            let node = TensorNode::new("exp2".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn softplus(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::softplus(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul(&input.sigmoid()?)?])
            });
            let node = TensorNode::new("softplus".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn recip(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::recip(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_clone = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let input_squared = input_clone.square()?;
                let neg_grad = grad_out.neg()?;
                Ok(vec![neg_grad.div(&input_squared)?])
            });
            let node = TensorNode::new("recip".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    // Comparison
    pub fn logical_not(&self) -> Result<Tensor> {
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::logical_not(
                    out_buf,
                    self.buffer(),
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }
}

impl Tensor {
    pub fn add_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = promote_tensor(self, target_dtype)?;
        let scalar = promote_scalar_for_tensor(scalar, target_dtype, self)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::add_scalar(
                    out_buf,
                    input.buffer(),
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.clone()])
            });
            let node = TensorNode::new("add_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sub_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = promote_tensor(self, target_dtype)?;
        let scalar = promote_scalar_for_tensor(scalar, target_dtype, self)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::sub_scalar(
                    out_buf,
                    input.buffer(),
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.clone()])
            });
            let node = TensorNode::new("sub_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn mul_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = promote_tensor(self, target_dtype)?;
        let scalar = promote_scalar_for_tensor(scalar, target_dtype, self)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::mul_scalar(
                    out_buf,
                    input.buffer(),
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let scalar_clone = scalar;
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul_scalar(scalar_clone)?])
            });
            let node = TensorNode::new("mul_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn div_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let input = promote_tensor(self, target_dtype)?;
        let scalar = promote_scalar_for_tensor(scalar, target_dtype, self)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::div_scalar(
                    out_buf,
                    input.buffer(),
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let scalar_clone = scalar;
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.div_scalar(scalar_clone)?])
            });
            let node = TensorNode::new("div_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn maximum_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = promote_tensor(self, target_dtype)?;
        let scalar = promote_scalar_for_tensor(scalar, target_dtype, self)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::maximum_scalar(
                    out_buf,
                    input.buffer(),
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let self_clone = self.clone();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let gt_mask = self_clone.gt_scalar(scalar)?;
                let eq_mask = self_clone.eq_scalar(scalar)?;
                let eq_half = eq_mask.mul_scalar(0.5)?;

                let grad_input = grad_out.mul(&gt_mask)?.add(&eq_half.mul(grad_out)?)?;

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("maximum_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn minimum_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = promote_tensor(self, target_dtype)?;
        let scalar = promote_scalar_for_tensor(scalar, target_dtype, self)?;

        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::minimum_scalar(
                    out_buf,
                    input.buffer(),
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let self_clone = self.clone();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let lt_mask = self_clone.lt_scalar(scalar)?;
                let eq_mask = self_clone.eq_scalar(scalar)?;
                let eq_half = eq_mask.mul_scalar(0.5)?;

                let grad_input = grad_out.mul(&lt_mask)?.add(&eq_half.mul(grad_out)?)?;

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("minimum_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn pow(&self, exponent: impl Into<Scalar>) -> Result<Tensor> {
        let exponent = exponent.into();
        let target_dtype = if exponent.is_float() && !exponent.is_integer_value() {
            exponent.dtype()
        } else {
            self.dtype()
        };

        let input = promote_tensor(self, target_dtype)?;
        let exponent = promote_scalar_for_tensor(exponent, target_dtype, self)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::pow(
                    out_buf,
                    input.buffer(),
                    exponent,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out
                    .mul_scalar(exponent)?
                    .mul(&input.pow(exponent.as_f32() - 1.0)?)?])
            });
            let node = TensorNode::new("pow".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn leaky_relu(&self, exponent: impl Into<Scalar>) -> Result<Tensor> {
        let exponent = exponent.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let input = promote_tensor(self, target_dtype)?;
        let exponent = promote_scalar_for_tensor(exponent, target_dtype, self)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::leaky_relu(
                    out_buf,
                    input.buffer(),
                    exponent,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let ones = input.gt_scalar(0.0)?.to_dtype(output.dtype())?;
                let alpha_mask = ones.mul_scalar(-1.0)?.add_scalar(1.0)?;

                let grad = grad_out.mul(&ones.add(&alpha_mask.mul_scalar(exponent)?)?)?;
                Ok(vec![grad])
            });
            let node = TensorNode::new("leaky_relu".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn elu(&self, exponent: impl Into<Scalar>) -> Result<Tensor> {
        let exponent = exponent.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let input = promote_tensor(self, target_dtype)?;
        let exponent = promote_scalar_for_tensor(exponent, target_dtype, self)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::elu(
                    out_buf,
                    input.buffer(),
                    exponent,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input = self.clone();
            let output = result.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let pos_mask = input.gt_scalar(0.0)?.to_dtype(output.dtype())?;
                let neg_mask = pos_mask.mul_scalar(-1.0)?.add_scalar(1.0)?;
                let neg_grad = output.mul(&neg_mask)?.add_scalar(exponent)?.mul(&neg_mask)?;
                let grad = grad_out.mul(&pos_mask.add(&neg_grad)?)?;

                Ok(vec![grad])
            });
            let node = TensorNode::new("elu".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    // Comparison operators
    pub fn eq_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let scalar = promote_scalar_for_tensor(scalar, self.dtype(), self)?;
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::eq_scalar(
                    out_buf,
                    self.buffer(),
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn ne_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let scalar = promote_scalar_for_tensor(scalar, self.dtype(), self)?;
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::ne_scalar(
                    out_buf,
                    self.buffer(),
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn lt_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let scalar = promote_scalar_for_tensor(scalar, self.dtype(), self)?;
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::lt_scalar(
                    out_buf,
                    self.buffer(),
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn le_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let scalar = promote_scalar_for_tensor(scalar, self.dtype(), self)?;
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::le_scalar(
                    out_buf,
                    self.buffer(),
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn gt_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let scalar = promote_scalar_for_tensor(scalar, self.dtype(), self)?;
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::gt_scalar(
                    out_buf,
                    self.buffer(),
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn ge_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let scalar = promote_scalar_for_tensor(scalar, self.dtype(), self)?;
        let mut result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::unary::ge_scalar(
                    out_buf,
                    self.buffer(),
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_metadata(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }
}

pub(crate) fn prepare_metadata(tensor: &Tensor) -> Vec<usize> {
    let mut info = Vec::new();

    // Add dimensions
    info.extend_from_slice(tensor.shape());
    // Add strides
    info.extend_from_slice(tensor.strides());
    // Add offset
    info.push(tensor.offset());

    info
}
