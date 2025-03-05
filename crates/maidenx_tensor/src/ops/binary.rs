use crate::{
    utils::{
        broadcast::broadcast_tensor,
        promotion::{get_promoted_dtype, promote_tensor},
    },
    Tensor, TensorNode,
};
use maidenx_core::{buffer::Buffer, dtype::DType, error::Result};

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::add(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_left = grad_out.sum_to_shape(&original_lhs_shape)?;
                let grad_right = grad_out.sum_to_shape(&original_rhs_shape)?;

                Ok(vec![grad_left, grad_right])
            });

            let node = TensorNode::new("add".to_string(), vec![lhs, rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::sub(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_left = grad_out.sum_to_shape(&original_lhs_shape)?;
                let grad_right = grad_out.neg()?.sum_to_shape(&original_rhs_shape)?;

                Ok(vec![grad_left, grad_right])
            });

            let node = TensorNode::new("sub".to_string(), vec![lhs, rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::mul(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let lhs_clone = lhs.clone();
            let rhs_clone = rhs.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_left = grad_out.mul(&rhs_clone)?.sum_to_shape(&original_lhs_shape)?;
                let grad_right = grad_out.mul(&lhs_clone)?.sum_to_shape(&original_rhs_shape)?;

                Ok(vec![grad_left, grad_right])
            });

            let node = TensorNode::new("mul".to_string(), vec![lhs, rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype().is_int() && rhs.dtype().is_int() {
            DType::F32
        } else if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::div(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let lhs_clone = lhs.clone();
            let rhs_clone = rhs.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_left = grad_out.div(&rhs_clone)?.sum_to_shape(&original_lhs_shape)?;
                let grad_right = grad_out
                    .mul(&lhs_clone)?
                    .div(&rhs_clone.square()?)?
                    .neg()?
                    .sum_to_shape(&original_rhs_shape)?;

                Ok(vec![grad_left, grad_right])
            });

            let node = TensorNode::new("div".to_string(), vec![lhs, rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn logical_and(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::logical_and(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn logical_or(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::logical_or(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn logical_xor(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::logical_xor(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn eq(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::eq(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn ne(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::ne(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn lt(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::lt(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn le(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::le(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn gt(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::gt(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn ge(&self, rhs: &Tensor) -> Result<Tensor> {
        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let lhs = broadcast_tensor(&lhs, &rhs)?;
        let rhs = broadcast_tensor(&rhs, &lhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let dims_and_strides = prepare_dims_and_strides(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::ge(out_buf, lhs.buffer(), rhs.buffer(), lhs.size(), lhs.ndim(), Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }
}

impl Tensor {
    pub fn add_(&mut self, rhs: &Tensor) -> Result<()> {
        let rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        let dims_and_strides = prepare_dims_and_strides(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|out_buf| {
                let lhs_in = &*out_buf as *const dyn Buffer;
                let lhs_out = &mut *out_buf as *mut dyn Buffer;
                maidenx_core::be::ops::binary::add(&mut *lhs_out, &*lhs_in, rhs.buffer(), size, ndim, Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(())
    }

    pub fn sub_(&mut self, rhs: &Tensor) -> Result<()> {
        let rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        let dims_and_strides = prepare_dims_and_strides(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|out_buf| {
                let lhs_in = &*out_buf as *const dyn Buffer;
                let lhs_out = &mut *out_buf as *mut dyn Buffer;
                maidenx_core::be::ops::binary::sub(&mut *lhs_out, &*lhs_in, rhs.buffer(), size, ndim, Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(())
    }

    pub fn mul_(&mut self, rhs: &Tensor) -> Result<()> {
        let rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        let dims_and_strides = prepare_dims_and_strides(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|out_buf| {
                let lhs_in = &*out_buf as *const dyn Buffer;
                let lhs_out = &mut *out_buf as *mut dyn Buffer;
                maidenx_core::be::ops::binary::mul(&mut *lhs_out, &*lhs_in, rhs.buffer(), size, ndim, Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(())
    }

    pub fn div_(&mut self, rhs: &Tensor) -> Result<()> {
        let rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        let dims_and_strides = prepare_dims_and_strides(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|out_buf| {
                let lhs_in = &*out_buf as *const dyn Buffer;
                let lhs_out = &mut *out_buf as *mut dyn Buffer;
                maidenx_core::be::ops::binary::div(&mut *lhs_out, &*lhs_in, rhs.buffer(), size, ndim, Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(())
    }
}

fn prepare_dims_and_strides(lhs: &Tensor, rhs: &Tensor) -> Vec<usize> {
    let mut dims_and_strides = Vec::new();

    // Add dimensions
    dims_and_strides.extend_from_slice(lhs.shape());

    // Add strides for both tensors
    dims_and_strides.extend_from_slice(lhs.strides());
    dims_and_strides.extend_from_slice(rhs.strides());

    dims_and_strides
}
