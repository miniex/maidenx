use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "tensor_reduce_ops")]
extern "C" {
    fn tensor_mean(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
    fn tensor_sum(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
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
}
