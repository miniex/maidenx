use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "tensor_shape_ops")]
extern "C" {
    fn tensor_transpose_2d(
        output: *mut f32,
        input: *const f32,
        rows: usize,
        cols: usize,
        stream: *mut c_void,
    );

    fn tensor_transpose_dim(
        output: *mut f32,
        input: *const f32,
        shape: *const usize,
        num_dims: i32,
        dim0: i32,
        dim1: i32,
        stream: *mut c_void,
    );
}

/// Transposes a 2D matrix on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_transpose_2d(
    output: *mut f32,
    input: *const f32,
    rows: usize,
    cols: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_transpose_2d(output, input, rows, cols, stream_handle);
    Ok(())
}

/// Transposes an n-dimensional tensor along specified dimensions on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires:
/// - Valid CUDA device pointers for `output` and `input`
/// - Valid shape array that matches the tensor dimensions
/// - Dimensions dim0 and dim1 must be valid indices into the shape array
/// - Properly sized and aligned buffers
/// - Valid CUDA stream if provided
pub unsafe fn cuda_tensor_transpose_dim(
    output: *mut f32,
    input: *const f32,
    shape: &[usize],
    dim0: i32,
    dim1: i32,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_transpose_dim(
        output,
        input,
        shape.as_ptr(),
        shape.len() as i32,
        dim0,
        dim1,
        stream_handle,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CudaBuffer, error::CudaResult};

    #[test]
    fn test_cuda_transpose_2d_with_stream() -> CudaResult<()> {
        let rows = 3;
        let cols = 4;
        let size = rows * cols;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        // Create test matrix: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        let input_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_transpose_2d(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                rows,
                cols,
                Some(&stream),
            )?;
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
    fn test_cuda_transpose_dim_with_stream() -> CudaResult<()> {
        let shape = vec![2, 3, 4]; // 2x3x4 tensor
        let size = shape.iter().product();
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        // Create test tensor with sequential values
        let input_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        // Transpose dimensions 1 and 2 (middle and last dimensions)
        unsafe {
            cuda_tensor_transpose_dim(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                &shape,
                1,
                2,
                Some(&stream),
            )?;
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
    fn test_cuda_transpose_dim_identity() -> CudaResult<()> {
        let shape = vec![2, 3, 4];
        let size = shape.iter().product();
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        // Transpose same dimension (should be identity operation)
        unsafe {
            cuda_tensor_transpose_dim(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                &shape,
                1,
                1,
                Some(&stream),
            )?;
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

