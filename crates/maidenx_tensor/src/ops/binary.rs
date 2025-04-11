use crate::{
    utils::{
        broadcast::broadcast_tensors,
        promotion::{get_promoted_dtype, promote_tensor},
    },
    Tensor, TensorNode,
};
use maidenx_core::{dtype::DType, error::Result};

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::add(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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

            let node = TensorNode::new("add".to_string(), vec![original_lhs, original_rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::sub(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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

            let node = TensorNode::new("sub".to_string(), vec![original_lhs, original_rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::mul(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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

            let node = TensorNode::new("mul".to_string(), vec![original_lhs, original_rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();
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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::div(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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

            let node = TensorNode::new("div".to_string(), vec![original_lhs, original_rhs], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn maximum(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::maximum(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

                Ok(())
            })?;
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let lhs_clone = lhs.clone();
            let rhs_clone = rhs.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let lhs_mask = lhs_clone.gt(&rhs_clone)?;
                let equal_mask = lhs_clone.eq(&rhs_clone)?;
                let equal_half = equal_mask.mul_scalar(0.5)?;

                let grad_left = grad_out
                    .mul(&lhs_mask)?
                    .add(&equal_half.mul(grad_out)?)?
                    .sum_to_shape(&original_lhs_shape)?;

                let rhs_mask = rhs_clone.gt(&lhs_clone)?;
                let grad_right = grad_out
                    .mul(&rhs_mask)?
                    .add(&equal_half.mul(grad_out)?)?
                    .sum_to_shape(&original_rhs_shape)?;

                Ok(vec![grad_left, grad_right])
            });

            let node = TensorNode::new(
                "maximum".to_string(),
                vec![original_lhs, original_rhs],
                Some(backward_fn),
            );
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn minimum(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();
        let original_lhs_shape = self.shape().to_vec();
        let original_rhs_shape = rhs.shape().to_vec();

        let target_dtype = if self.dtype() != rhs.dtype() {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        } else {
            self.dtype()
        };

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), lhs.dtype())?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::minimum(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

                Ok(())
            })?;
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let lhs_clone = lhs.clone();
            let rhs_clone = rhs.clone();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let lhs_mask = lhs_clone.lt(&rhs_clone)?;
                let equal_mask = lhs_clone.eq(&rhs_clone)?;
                let equal_half = equal_mask.mul_scalar(0.5)?;

                let grad_left = grad_out
                    .mul(&lhs_mask)?
                    .add(&equal_half.mul(grad_out)?)?
                    .sum_to_shape(&original_lhs_shape)?;

                let rhs_mask = rhs_clone.lt(&lhs_clone)?;
                let grad_right = grad_out
                    .mul(&rhs_mask)?
                    .add(&equal_half.mul(grad_out)?)?
                    .sum_to_shape(&original_rhs_shape)?;

                Ok(vec![grad_left, grad_right])
            });

            let node = TensorNode::new(
                "minimum".to_string(),
                vec![original_lhs, original_rhs],
                Some(backward_fn),
            );
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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::logical_and(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::logical_or(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::logical_xor(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::eq(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::ne(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::lt(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::le(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::gt(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

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
        let (lhs, rhs) = broadcast_tensors(&lhs, &rhs)?;

        let mut result = Self::empty_with_spec(lhs.shape(), lhs.device(), DType::BOOL)?;

        let metadata = prepare_metadata(&lhs, &rhs);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::binary::ge(
                    out_buf,
                    lhs.buffer(),
                    rhs.buffer(),
                    lhs.size(),
                    lhs.ndim(),
                    Some(&metadata),
                )?;

                Ok(())
            })?;
        }

        Ok(result)
    }
}

impl Tensor {
    pub fn add_(&mut self, rhs: &Tensor) -> Result<()> {
        let mut rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        if self.dtype() != rhs.dtype() {
            rhs.with_dtype(self.dtype())?;
        }

        let metadata = prepare_metadata(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|self_buf| {
                maidenx_core::be::ops::binary::add_inplace(self_buf, rhs.buffer(), size, ndim, Some(&metadata))?;
                Ok(())
            })?;
        }

        Ok(())
    }

    pub fn sub_(&mut self, rhs: &Tensor) -> Result<()> {
        let mut rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        if self.dtype() != rhs.dtype() {
            rhs.with_dtype(self.dtype())?;
        }

        let metadata = prepare_metadata(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|self_buf| {
                maidenx_core::be::ops::binary::sub_inplace(self_buf, rhs.buffer(), size, ndim, Some(&metadata))?;
                Ok(())
            })?;
        }

        Ok(())
    }

    pub fn mul_(&mut self, rhs: &Tensor) -> Result<()> {
        let mut rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        if self.dtype() != rhs.dtype() {
            rhs.with_dtype(self.dtype())?;
        }

        let metadata = prepare_metadata(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|self_buf| {
                maidenx_core::be::ops::binary::mul_inplace(self_buf, rhs.buffer(), size, ndim, Some(&metadata))?;
                Ok(())
            })?;
        }

        Ok(())
    }

    pub fn div_(&mut self, rhs: &Tensor) -> Result<()> {
        let mut rhs = if self.shape() == rhs.shape() {
            rhs.clone()
        } else {
            rhs.broadcast(self.shape())?
        };

        if self.dtype() != rhs.dtype() {
            rhs.with_dtype(self.dtype())?;
        }

        let metadata = prepare_metadata(self, &rhs);
        let size = self.size();
        let ndim = self.ndim();

        unsafe {
            self.with_buffer_mut(|self_buf| {
                maidenx_core::be::ops::binary::div_inplace(self_buf, rhs.buffer(), size, ndim, Some(&metadata))?;
                Ok(())
            })?;
        }

        Ok(())
    }
}

fn prepare_metadata(lhs: &Tensor, rhs: &Tensor) -> Vec<usize> {
    let mut metadata = Vec::new();

    // Add dimensions
    metadata.extend_from_slice(lhs.shape());

    // Add strides for both tensors
    metadata.extend_from_slice(lhs.strides());
    metadata.extend_from_slice(rhs.strides());

    // Add offsets for both tensors
    metadata.push(lhs.offset());
    metadata.push(rhs.offset());

    metadata
}
