pub trait TensorAdapter {
    fn to_flat_vec(self) -> Vec<f32>;
    fn to_shape(&self) -> Vec<usize>;
}

impl TensorAdapter for f32 {
    fn to_flat_vec(self) -> Vec<f32> {
        vec![self]
    }

    fn to_shape(&self) -> Vec<usize> {
        vec![1]
    }
}

impl TensorAdapter for Vec<f32> {
    fn to_flat_vec(self) -> Vec<f32> {
        self
    }

    fn to_shape(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

impl TensorAdapter for Vec<Vec<f32>> {
    fn to_flat_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|inner| inner.into_iter())
            .collect()
    }

    fn to_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            vec![0, 0]
        } else {
            vec![self.len(), self[0].len()]
        }
    }
}

impl TensorAdapter for Vec<Vec<Vec<f32>>> {
    fn to_flat_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|matrix| matrix.into_iter().flat_map(|row| row.into_iter()))
            .collect()
    }

    fn to_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            vec![0, 0, 0]
        } else if self[0].is_empty() {
            vec![self.len(), 0, 0]
        } else {
            vec![self.len(), self[0].len(), self[0][0].len()]
        }
    }
}

impl TensorAdapter for Vec<Vec<Vec<Vec<f32>>>> {
    fn to_flat_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|cube| {
                cube.into_iter()
                    .flat_map(|matrix| matrix.into_iter().flat_map(|row| row.into_iter()))
            })
            .collect()
    }

    fn to_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            vec![0, 0, 0, 0]
        } else if self[0].is_empty() {
            vec![self.len(), 0, 0, 0]
        } else if self[0][0].is_empty() {
            vec![self.len(), self[0].len(), 0, 0]
        } else {
            vec![
                self.len(),
                self[0].len(),
                self[0][0].len(),
                self[0][0][0].len(),
            ]
        }
    }
}
