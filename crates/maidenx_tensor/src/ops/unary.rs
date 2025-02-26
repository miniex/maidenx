use crate::{utils::promotion::promote_tensor, Tensor, TensorNode};
use maidenx_core::{dtype::DType, error::Result, scalar::Scalar};

impl Tensor {
    pub fn neg(&self) -> Result<Tensor> {
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::neg(out_buf, &*self.buffer()?, self.size(), self.ndim(), Some(&prepare_dims_and_strides(self)))?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.neg()?]) });
            let node = TensorNode::new("neg".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn abs(&self) -> Result<Tensor> {
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::abs(out_buf, &*self.buffer()?, self.size(), self.ndim(), Some(&prepare_dims_and_strides(self)))?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.mul(&input.sign()?)?]) });
            let node = TensorNode::new("abs".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sign(&self) -> Result<Tensor> {
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::sign(out_buf, &*self.buffer()?, self.size(), self.ndim(), Some(&prepare_dims_and_strides(self)))?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], _grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![]) });
            let node = TensorNode::new("sign".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn square(&self) -> Result<Tensor> {
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::square(out_buf, &*self.buffer()?, self.size(), self.ndim(), Some(&prepare_dims_and_strides(self)))?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.mul_scalar(2.0)?.mul(&input)?]) });
            let node = TensorNode::new("square".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::sqrt(
                    out_buf,
                    &*input.buffer()?,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_dims_and_strides(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let output = result.clone();
            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.div(&output.mul_scalar(2.0)?)?]) });
            let node = TensorNode::new("sqrt".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn relu(&self) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::relu(
                    out_buf,
                    &*input.buffer()?,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_dims_and_strides(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
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
        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::sigmoid(
                    out_buf,
                    &*input.buffer()?,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_dims_and_strides(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
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
        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::tanh(
                    out_buf,
                    &*input.buffer()?,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_dims_and_strides(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
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

    pub fn logical_not(&self) -> Result<Tensor> {
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::logical_not(
                    out_buf,
                    &*self.buffer()?,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
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
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::add_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.clone()]) });
            let node = TensorNode::new("add_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sub_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::sub_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.clone()]) });
            let node = TensorNode::new("sub_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn mul_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let mut result = Self::empty_with_spec(self.shape(), self.device(), self.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::mul_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.mul_scalar(scalar)?]) });
            let node = TensorNode::new("mul_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn div_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::div_scalar(
                    out_buf,
                    &*input.buffer()?,
                    scalar,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_dims_and_strides(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.div_scalar(scalar)?]) });
            let node = TensorNode::new("div_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn pow(&self, exponent: impl Into<Scalar>) -> Result<Tensor> {
        let exponent = exponent.into();
        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };

        let input = promote_tensor(self, target_dtype)?;
        let mut result = Self::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::pow(
                    out_buf,
                    &*input.buffer()?,
                    exponent,
                    input.size(),
                    input.ndim(),
                    Some(&prepare_dims_and_strides(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad {
            result.with_grad()?;

            let input = self.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.mul_scalar(exponent)?.mul(&input.pow(exponent.as_f32() - 1.0)?)?])
            });
            let node = TensorNode::new("pow".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    // Comparison operators
    pub fn eq_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::eq_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn ne_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::ne_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn lt_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::lt_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn le_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::le_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn gt_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::gt_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn ge_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor> {
        let scalar = scalar.into();
        let result = Self::empty_with_spec(self.shape(), self.device(), DType::BOOL)?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::buffer::ops::unary::ge_scalar(
                    out_buf,
                    &*self.buffer()?,
                    scalar,
                    self.size(),
                    self.ndim(),
                    Some(&prepare_dims_and_strides(self)),
                )?;
                Ok(())
            })?;
        }

        Ok(result)
    }
}

fn prepare_dims_and_strides(tensor: &Tensor) -> Vec<usize> {
    let mut info = Vec::new();

    // Add dimensions
    info.extend_from_slice(tensor.shape());
    // Add strides
    info.extend_from_slice(tensor.strides());

    info
}
