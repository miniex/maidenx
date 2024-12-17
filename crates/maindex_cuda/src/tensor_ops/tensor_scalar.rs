use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "tensor_scalar_ops")]
extern "C" {
    fn tensor_scalar_add(
        output: *mut f32,
        input: *const f32,
        scalar: f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_scalar_sub(
        output: *mut f32,
        input: *const f32,
        scalar: f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_scalar_mul(
        output: *mut f32,
        input: *const f32,
        scalar: f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_scalar_div(
        output: *mut f32,
        input: *const f32,
        scalar: f32,
        size: usize,
        stream: *mut c_void,
    );
}

/// Adds a scalar to each element of a tensor on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_scalar_add(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_scalar_add(output, input, scalar, size, stream_handle);
    Ok(())
}

/// Subtracts a scalar from each element of a tensor on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_scalar_sub(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_scalar_sub(output, input, scalar, size, stream_handle);
    Ok(())
}

/// Multiplies each element of a tensor by a scalar on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
pub unsafe fn cuda_tensor_scalar_mul(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_scalar_mul(output, input, scalar, size, stream_handle);
    Ok(())
}

/// Divides each element of a tensor by a scalar on the CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires valid CUDA device pointers for `output` and `input`,
/// properly sized and aligned buffers, and a valid CUDA stream if provided.
/// Also, the scalar must not be zero to avoid division by zero.
pub unsafe fn cuda_tensor_scalar_div(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_scalar_div(output, input, scalar, size, stream_handle);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CudaBuffer, error::CudaResult};

    #[test]
    fn test_cuda_scalar_add_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![1.0; size];
        let scalar = 2.0;

        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_scalar_add(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                scalar,
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                3.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_cuda_scalar_sub_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![3.0; size];
        let scalar = 1.0;

        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_scalar_sub(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                scalar,
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 2.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                2.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_cuda_scalar_mul_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![3.0; size];
        let scalar = 2.0;

        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_scalar_mul(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                scalar,
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 6.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                6.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_cuda_scalar_div_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        let input_data: Vec<f32> = vec![6.0; size];
        let scalar = 2.0;

        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_scalar_div(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                scalar,
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                3.0,
                val
            );
        }

        Ok(())
    }
}
