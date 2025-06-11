pub struct TensorIndexIterator {
    pub shape: Vec<usize>,
    pub current: Vec<usize>,
    pub done: bool,
}

impl Iterator for TensorIndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        let mut dim = self.current.len() - 1;
        loop {
            self.current[dim] += 1;
            if self.current[dim] < self.shape[dim] {
                break;
            }

            self.current[dim] = 0;

            if dim == 0 {
                self.done = true;
                break;
            }

            dim -= 1;
        }

        Some(result)
    }
}