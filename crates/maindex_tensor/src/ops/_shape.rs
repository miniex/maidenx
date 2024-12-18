// Broadcast Like

use crate::{
    error::{TensorError, TensorResult},
    Tensor,
};

impl Tensor {
    pub fn broadcast_like(&self, target: &Tensor) -> TensorResult<Self> {
        let self_shape = &self.shape;
        let target_shape = &target.shape;

        // 1. Pad shorter shape with ones on the left
        let rank_diff = target_shape.len().saturating_sub(self_shape.len());
        let mut padded_self_shape = vec![1; rank_diff];
        padded_self_shape.extend(self_shape.iter());

        // 2. Check broadcasting compatibility
        for (i, (&a, &b)) in padded_self_shape
            .iter()
            .zip(target_shape.iter())
            .enumerate()
        {
            if a != b && a != 1 && b != 1 {
                return Err(TensorError::InvalidShape {
                    reason: format!(
                        "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                        self_shape, target_shape, i
                    ),
                });
            }
        }

        // 3. Create new data with broadcasting
        let original_data = self.to_vec()?;
        let mut new_data = Vec::with_capacity(target_shape.iter().product());

        // 4. Calculate strides for both old and new shapes
        let original_strides = self.strides.clone();
        let target_size = target_shape.iter().product();
        let mut indices = vec![0; target_shape.len()];

        for _ in 0..target_size {
            // Calculate original data index using strides
            let mut original_index = 0;
            (rank_diff..target_shape.len()).for_each(|i| {
                let self_idx = i - rank_diff;
                if self_idx < self_shape.len() {
                    let dim_size = self_shape[self_idx];
                    let idx = if dim_size == 1 { 0 } else { indices[i] };
                    original_index += idx * original_strides[self_idx];
                }
            });

            new_data.push(original_data[original_index]);

            // Update indices
            for i in (0..target_shape.len()).rev() {
                indices[i] += 1;
                if indices[i] < target_shape[i] {
                    break;
                }
                indices[i] = 0;
            }
        }

        // 5. Create new tensor with broadcasted data
        let result = Tensor::from_vec_with_device(new_data, target_shape, &self.device)?;

        Ok(Tensor {
            buffer: result.buffer,
            shape: result.shape,
            strides: result.strides,
            device: result.device,
            requires_grad: self.requires_grad,
            node: self.node.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_device::Device;

    #[test]
    fn test_broadcast_like() {
        let device = Device::cpu();

        // Case 1: [3] -> [1, 3]
        let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device).unwrap();
        let b = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]], &device).unwrap();

        let broadcasted = a.broadcast_like(&b).unwrap();
        assert_eq!(broadcasted.shape(), &[1, 3]);
        assert_eq!(broadcasted.strides(), &[3, 1]);
        assert_eq!(broadcasted.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        // Case 2: [1, 3] -> [2, 3]
        let c = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]], &device).unwrap();
        let d = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]; 2], &device).unwrap();

        let broadcasted_c = c.broadcast_like(&d).unwrap();
        assert_eq!(broadcasted_c.shape(), &[2, 3]);
        assert_eq!(broadcasted_c.strides(), &[3, 1]);
        assert_eq!(
            broadcasted_c.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    // basic_ops
    #[test]
    fn test_broadcast_add() -> TensorResult<()> {
        let device = Device::cpu();

        // Case 1
        let tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let tensor2 = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let result = tensor1.add(&tensor2)?;
        assert_eq!(result.to_vec()?, vec![5.0, 7.0, 9.0]);

        // Case 2
        let tensor1d = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
        let tensor2d =
            Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;
        let result = tensor1d.add(&tensor2d)?;
        assert_eq!(result.to_vec()?, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);

        // Case 3
        let a = Tensor::from_device(vec![1.0, 2.0], &device)?.reshape(&[2, 1])?;
        let b = Tensor::from_device(vec![3.0, 4.0, 5.0], &device)?.reshape(&[1, 3])?;
        let result = a.add(&b)?;
        assert_eq!(result.to_vec()?, vec![4.0, 5.0, 6.0, 5.0, 6.0, 7.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_sub() -> TensorResult<()> {
        let device = Device::cpu();

        // Case 1
        let tensor1 = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let tensor2 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let result = tensor1.sub(&tensor2)?;
        assert_eq!(result.to_vec()?, vec![3.0, 3.0, 3.0]);

        // Case 2
        let tensor1d = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
        let tensor2d =
            Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;
        let result = tensor1d.sub(&tensor2d)?;
        assert_eq!(result.to_vec()?, vec![0.0, 0.0, 0.0, -3.0, -3.0, -3.0]);

        // Case 3
        let a = Tensor::from_device(vec![1.0, 2.0], &device)?.reshape(&[2, 1])?;
        let b = Tensor::from_device(vec![3.0, 4.0, 5.0], &device)?.reshape(&[1, 3])?;
        let result = a.sub(&b)?;
        assert_eq!(result.to_vec()?, vec![-2.0, -3.0, -4.0, -1.0, -2.0, -3.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_mul() -> TensorResult<()> {
        let device = Device::cpu();

        // Case 1: Same dimension vectors [3] * [3]
        let tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let tensor2 = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        let result = tensor1.mul(&tensor2)?;
        assert_eq!(result.to_vec()?, vec![2.0, 6.0, 12.0]);

        // Case 2: 1D to 2D with reshape [1,2] * [2,2]
        let vector = Tensor::from_device(vec![2.0, 3.0], &device)?.reshape(&[1, 2])?;
        let matrix = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let result = vector.mul(&matrix)?;
        assert_eq!(result.to_vec()?, vec![2.0, 6.0, 6.0, 12.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_div() -> TensorResult<()> {
        let device = Device::cpu();

        // Case 1: Same dimension vectors [3] / [3]
        let tensor1 = Tensor::from_device(vec![2.0, 4.0, 6.0], &device)?;
        let tensor2 = Tensor::from_device(vec![2.0, 2.0, 2.0], &device)?;
        let result = tensor1.div(&tensor2)?;
        assert_eq!(result.to_vec()?, vec![1.0, 2.0, 3.0]);

        // Case 2: Broadcasting with reshape [2,2] / [1,2]
        let matrix = Tensor::from_device(vec![vec![2.0, 4.0], vec![6.0, 8.0]], &device)?;
        let vector = Tensor::from_device(vec![2.0, 2.0], &device)?.reshape(&[1, 2])?;
        let result = matrix.div(&vector)?;
        assert_eq!(result.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_add_backward() -> TensorResult<()> {
        let device = Device::cpu();

        // Test case: [1,3] + [2,3]
        let mut tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
        let mut tensor2 =
            Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;

        tensor1.with_grad();
        tensor2.with_grad();

        let result = tensor1.add(&tensor2)?;
        result.backward()?;

        // Check gradients
        let tensor1_grad = tensor1.grad()?.unwrap().to_vec()?;
        assert_eq!(tensor1_grad, vec![2.0, 2.0, 2.0]); // Sum across broadcast dimension

        let tensor2_grad = tensor2.grad()?.unwrap().to_vec()?;
        assert_eq!(tensor2_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_mul_backward() -> TensorResult<()> {
        let device = Device::cpu();

        // Test case: [1,3] + [2,3]
        let mut tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
        let mut tensor2 =
            Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;

        tensor1.with_grad();
        tensor2.with_grad();

        let result = tensor1.mul(&tensor2)?;
        result.backward()?;

        // Check gradients
        let tensor1_grad = tensor1.grad()?.unwrap().to_vec()?;
        let tensor2_grad = tensor2.grad()?.unwrap().to_vec()?;

        // Verify gradient shapes and values
        assert_eq!(tensor1_grad.len(), 3);
        assert_eq!(tensor2_grad.len(), 6);

        Ok(())
    }

    #[test]
    fn test_broadcast_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();

        // Setup tensors with compatible broadcasting shapes
        let mut tensor1 = Tensor::from_device(vec![1.0, 2.0], &device)?.reshape(&[1, 2])?;
        let mut tensor2 = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let mut tensor3 = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

        tensor1.with_grad();
        tensor2.with_grad();
        tensor3.with_grad();

        // (tensor1 * tensor2) + tensor3 where tensor1 is [1,2], tensor2 and tensor3 are [2,2]
        let result = tensor1.mul(&tensor2)?.add(&tensor3)?;
        result.backward()?;

        // Verify gradient shapes
        assert_eq!(tensor1.grad()?.unwrap().to_vec()?.len(), 2);
        assert_eq!(tensor2.grad()?.unwrap().to_vec()?.len(), 4);
        assert_eq!(tensor3.grad()?.unwrap().to_vec()?.len(), 4);

        Ok(())
    }

    #[test]
    fn test_broadcast_failure() -> TensorResult<()> {
        let device = Device::cpu();

        // Case 1: Incompatible broadcasting shapes
        let tensor1 = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let tensor2 = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]], &device)?;
        assert!(tensor1.add(&tensor2).is_err());

        // Case 2: Cannot broadcast larger dimension to smaller
        let tensor3 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let tensor4 = Tensor::from_device(vec![1.0, 2.0], &device)?;
        assert!(tensor3.add(&tensor4).is_err());

        Ok(())
    }
}
