use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "tensor_reduce_ops")]
extern "C" {
    fn tensor_mean(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
    fn tensor_sum(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
    fn tensor_sum_with_dim(
        output: *mut f32,
        input: *const f32,
        input_shape: *const i32,
        num_dims: i32,
        reduction_dim: i32,
        stream: *mut c_void,
    );
}

/// Calculates the mean of all elements in a tensor on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_mean(
    output: *mut f32,
    input: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_mean(output, input, size, stream_handle);
    Ok(())
}

/// Calculates the sum of all elements in a tensor on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_sum(
    output: *mut f32,
    input: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_sum(output, input, size, stream_handle);
    Ok(())
}

/// Calculates the sum along a specified dimension in a tensor on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, valid shape information, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_sum_with_dim(
    output: *mut f32,
    input: *const f32,
    input_shape: &[i32],
    reduction_dim: i32,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_sum_with_dim(
        output,
        input,
        input_shape.as_ptr(),
        input_shape.len() as i32,
        reduction_dim,
        stream_handle,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CudaBuffer, error::CudaResult};

    #[test]
    fn test_cuda_mean_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(1, 0)?; // Single output for mean
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![2.0; size];
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_mean(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        assert!(
            (output_data[0] - 2.0).abs() < 1e-5,
            "Mean mismatch: expected {}, got {}",
            2.0,
            output_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_cuda_sum_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(1, 0)?; // Single output for sum
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![2.0; size];
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_sum(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        let expected_sum = 2.0 * (size as f32);
        assert!(
            (output_data[0] - expected_sum).abs() < 1e-5,
            "Sum mismatch: expected {}, got {}",
            expected_sum,
            output_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_cuda_mean_zero_elements() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(1, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![0.0; size];
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_mean(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        assert!(
            output_data[0].abs() < 1e-5,
            "Mean of zeros mismatch: expected {}, got {}",
            0.0,
            output_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_cuda_sum_zero_elements() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(1, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![0.0; size];
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_sum(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        assert!(
            output_data[0].abs() < 1e-5,
            "Sum of zeros mismatch: expected {}, got {}",
            0.0,
            output_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_cuda_sum_with_dim() -> CudaResult<()> {
        // 2x3x4 tensor test
        let input_shape = vec![2, 3, 4];
        let size = input_shape.iter().product::<i32>() as usize;

        let mut input_data = Vec::with_capacity(size);
        for i in 0..size {
            input_data.push(i as f32);
        }

        let mut input_buf = CudaBuffer::new(size, 0)?;
        input_buf.copy_from_host(&input_data)?;

        // Reduction along dim=1 (middle dimension)
        let reduction_dim = 1;
        let output_size = size / input_shape[reduction_dim as usize] as usize;
        let mut output_buf = CudaBuffer::new(output_size, 0)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_sum_with_dim(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                &input_shape,
                reduction_dim,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; output_size];
        output_buf.copy_to_host(&mut output_data)?;

        // Calculate expected results
        let mut expected = vec![0.0f32; output_size];
        for i in 0..2 {
            // First dimension
            for j in 0..4 {
                // Last dimension
                let mut sum = 0.0;
                for k in 0..3 {
                    // Middle dimension (being reduced)
                    let idx = i * 12 + k * 4 + j;
                    sum += input_data[idx];
                }
                expected[i * 4 + j] = sum;
            }
        }

        // Verify results
        for i in 0..output_size {
            assert!(
                (output_data[i] - expected[i]).abs() < 1e-5,
                "Sum with dim mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                output_data[i]
            );
        }

        Ok(())
    }
}
