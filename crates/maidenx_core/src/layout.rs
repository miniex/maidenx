use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: &[usize], strides: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset: 0,
        }
    }

    pub fn from_shape(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: 0,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn set_shape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }
    pub fn set_strides(&mut self, strides: &[usize]) {
        self.strides = strides.to_vec();
    }
    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn dim_size(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn view(&mut self, new_shape: &[usize]) -> Result<()> {
        let old_size = self.size();
        let new_size = new_shape.iter().product();

        if old_size != new_size {
            return Err(Error::IncompatibleShape(format!(
                "Cannot reshape layout of size {} to size {}",
                old_size, new_size
            )));
        }

        self.shape = new_shape.to_vec();
        self.strides = Self::compute_strides(new_shape);

        Ok(())
    }

    pub fn transpose(&mut self, dim0: usize, dim1: usize) -> Result<()> {
        // Bounds check
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Ok(());
        }

        // Swap shape and strides
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);

        Ok(())
    }

    pub fn slice(&self, dim: usize, start: isize, end: Option<isize>, step: isize) -> Result<Self> {
        // Check dimension bounds
        if dim >= self.ndim() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        if step == 0 {
            return Err(Error::InvalidArgument("Slice step cannot be zero".to_string()));
        }

        let dim_size = self.shape[dim] as isize;

        // Normalize negative indices and handle None for end
        let start_idx = if start < 0 { dim_size + start } else { start };
        let end_idx = match end {
            Some(e) => {
                if e < 0 {
                    dim_size + e
                } else {
                    e
                }
            }
            None => {
                if step > 0 {
                    dim_size
                } else {
                    -1
                }
            }
        };

        // Clamp indices to valid range
        let start_idx = start_idx.clamp(0, dim_size);
        let end_idx = end_idx.clamp(if step > 0 { start_idx } else { -1 }, dim_size);

        // Calculate new size for the dimension
        let new_size = if step > 0 {
            (end_idx - start_idx + step - 1) / step
        } else {
            (start_idx - end_idx + (-step) - 1) / (-step)
        };

        if new_size <= 0 {
            return Err(Error::InvalidShape {
                message: "Slice results in empty tensor".to_string(),
            });
        }

        // Create a new Layout with adjusted shape, strides and offset
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape[dim] = new_size as usize;
        new_strides[dim] *= step.unsigned_abs();

        // Calculate new offset
        let new_offset = self.offset + (start_idx as usize * self.strides[dim]);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }

    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Result<Self> {
        // Check dimension bounds
        if dim >= self.ndim() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        if size == 0 {
            return Err(Error::InvalidArgument("Unfold size cannot be zero".to_string()));
        }

        if step == 0 {
            return Err(Error::InvalidArgument("Unfold step cannot be zero".to_string()));
        }

        let dim_size = self.shape[dim];

        // Calculate number of windows
        let n_windows = if dim_size >= size { (dim_size - size) / step + 1 } else { 0 };

        if n_windows == 0 {
            return Err(Error::InvalidShape {
                message: "Unfold results in empty tensor".to_string(),
            });
        }

        // Create new shape and strides
        let mut new_shape = Vec::with_capacity(self.ndim() + 1);
        let mut new_strides = Vec::with_capacity(self.ndim() + 1);

        // Copy shape and strides up to dim
        new_shape.extend_from_slice(&self.shape[..dim]);
        new_strides.extend_from_slice(&self.strides[..dim]);

        // Add window count for the original dimension
        new_shape.push(n_windows);
        new_strides.push(self.strides[dim] * step);

        // Add the window size as a new dimension
        new_shape.push(size);
        new_strides.push(self.strides[dim]);

        // Copy remaining dimensions
        new_shape.extend_from_slice(&self.shape[dim + 1..]);
        new_strides.extend_from_slice(&self.strides[dim + 1..]);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    // helper

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        // Handle scalar case (empty shape)
        if shape.is_empty() {
            return vec![];
        }

        // Original logic for non-scalar tensors
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn compute_size(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    pub fn can_broadcast_like(&self, target: &Layout) -> bool {
        let self_shape = &self.shape;
        let target_shape = &target.shape;

        // 1. Pad the shorter shape with ones on the left
        let rank_diff = target_shape.len().saturating_sub(self_shape.len());
        let mut padded_self_shape = vec![1; rank_diff];
        padded_self_shape.extend(self_shape.iter());

        // 2. Check broadcasting compatibility
        for (&a, &b) in padded_self_shape.iter().zip(target_shape.iter()) {
            if a != b && a != 1 && b != 1 {
                return false; // Shapes are not compatible
            }
        }

        true
    }
}
