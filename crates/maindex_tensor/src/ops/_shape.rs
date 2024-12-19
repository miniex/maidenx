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
