#[inline]
pub fn is_contiguous(num_dims: usize, dims: &[usize], strides: &[usize]) -> bool {
    let mut acc = 1;
    for d in 0..num_dims {
        let dim_idx = num_dims - 1 - d;
        if dims[dim_idx] > 1 && acc != strides[dim_idx] {
            return false;
        }
        acc *= dims[dim_idx];
    }
    true
}

#[inline]
pub fn get_strided_index(idx: usize, num_dims: usize, dims: &[usize], strides: &[usize]) -> usize {
    let mut strided_i = 0;
    let mut remaining_idx = idx;

    for d in 0..num_dims {
        let dim_idx = num_dims - 1 - d;
        strided_i += (remaining_idx % dims[dim_idx]) * strides[dim_idx];
        remaining_idx /= dims[dim_idx];
    }

    strided_i
}

#[inline]
pub fn restrided(strided_i: usize, num_dims: usize, dims: &[usize], strides: &[usize], new_strides: &[usize]) -> usize {
    let mut idx = 0;

    for d in 0..num_dims {
        idx += if strides[d] == 0 {
            0
        } else {
            ((strided_i / strides[d]) % dims[d]) * new_strides[d]
        };
    }

    idx
}
