use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Layout {
    pub fn new(shape: &[usize], strides: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        }
    }

    pub fn from_shape(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        }
    }

    pub fn set_shape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }
    pub fn set_strides(&mut self, strides: &[usize]) {
        self.strides = strides.to_vec();
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn size_dim(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
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
