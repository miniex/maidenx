use crate::error::CpuResult;

/// Performs 2D matrix transpose on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Both buffers must have size >= `rows * cols`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_transpose_2d(
    output: *mut f32,
    input: *const f32,
    rows: usize,
    cols: usize,
) -> CpuResult<()> {
    for i in 0..rows {
        for j in 0..cols {
            *output.add(j * rows + i) = *input.add(i * cols + j);
        }
    }
    Ok(())
}

/// Helper function to calculate strides for a given shape
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Performs n-dimensional tensor transpose along specified dimensions on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Both buffers must have size >= product of all dimensions in shape
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
/// * dim0 and dim1 must be valid indices into shape
/// * shape must accurately describe the tensor dimensions
pub unsafe fn cpu_tensor_transpose_dim(
    output: *mut f32,
    input: *const f32,
    shape: &[usize],
    dim0: i32,
    dim1: i32,
) -> CpuResult<()> {
    // Validate dimensions
    if dim0 < 0 || dim0 >= shape.len() as i32 || dim1 < 0 || dim1 >= shape.len() as i32 {
        return Ok(()); // Early return for invalid dimensions
    }

    let num_dims = shape.len();
    let total_size: usize = shape.iter().product();

    // Calculate strides for input tensor
    let input_strides = calculate_strides(shape);

    // Create output shape (with swapped dimensions)
    let mut output_shape = shape.to_vec();
    output_shape.swap(dim0 as usize, dim1 as usize);
    let output_strides = calculate_strides(&output_shape);

    for idx in 0..total_size {
        // Calculate coordinates for current index
        let mut coords = Vec::with_capacity(num_dims);
        let mut remaining = idx;
        for &stride in &input_strides {
            coords.push(remaining / stride);
            remaining %= stride;
        }

        // Swap coordinates for the specified dimensions
        coords.swap(dim0 as usize, dim1 as usize);

        // Calculate output index
        let mut output_idx = 0;
        for (i, &coord) in coords.iter().enumerate() {
            output_idx += coord * output_strides[i];
        }

        // Perform transpose
        *output.add(output_idx) = *input.add(idx);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CpuBuffer, error::CpuResult};

    #[test]
    fn test_cpu_transpose_2d() -> CpuResult<()> {
        let rows = 3;
        let cols = 4;
        let size = rows * cols;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        // Input matrix: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        let input_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_transpose_2d(output_buf.as_mut_ptr(), input_buf.as_ptr(), rows, cols)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        // Expected: [[1,5,9], [2,6,10], [3,7,11], [4,8,12]]
        let expected: Vec<f32> = vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ];

        for (i, (&val, &exp)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_cpu_transpose_dim() -> CpuResult<()> {
        let shape = vec![2, 3, 4]; // 2x3x4 tensor
        let size = shape.iter().product();
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        // Create test tensor with sequential values
        let input_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        // Transpose dimensions 1 and 2 (middle and last dimensions)
        unsafe {
            cpu_tensor_transpose_dim(output_buf.as_mut_ptr(), input_buf.as_ptr(), &shape, 1, 2)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        // Helper function to calculate index in original tensor
        let calc_original_idx = |i: usize, j: usize, k: usize| -> usize {
            i * (shape[1] * shape[2]) + j * shape[2] + k
        };

        // Helper function to calculate index in transposed tensor (dims 1,2 swapped)
        let calc_transposed_idx = |i: usize, j: usize, k: usize| -> usize {
            i * (shape[2] * shape[1]) + k * shape[1] + j
        };

        // Verify the transpose
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let orig_idx = calc_original_idx(i, j, k);
                    let trans_idx = calc_transposed_idx(i, j, k);
                    assert!(
                        (output_data[trans_idx] - input_data[orig_idx]).abs() < 1e-5,
                        "Mismatch at position [{},{},{}]: expected {}, got {}",
                        i,
                        j,
                        k,
                        input_data[orig_idx],
                        output_data[trans_idx]
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_cpu_transpose_dim_identity() -> CpuResult<()> {
        let shape = vec![2, 3, 4];
        let size = shape.iter().product();
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        // Transpose same dimension (should be identity operation)
        unsafe {
            cpu_tensor_transpose_dim(output_buf.as_mut_ptr(), input_buf.as_ptr(), &shape, 1, 1)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        // Verify that the output matches the input (identity transpose)
        for (i, (&val, &exp)) in output_data.iter().zip(input_data.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        Ok(())
    }
}

