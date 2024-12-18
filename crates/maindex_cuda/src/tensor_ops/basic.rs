use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "tensor_basic_ops")]
extern "C" {
    fn tensor_add(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_sub(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_mul(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_div(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        size: usize,
        stream: *mut c_void,
    );
    fn tensor_mat_mul(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    );
}

/// Performs element-wise addition of two tensors on CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires that:
/// * `output`, `input1`, and `input2` must be valid CUDA device pointers
/// * `output` must point to a buffer with size >= `size`
/// * `input1` and `input2` must point to buffers with size >= `size`
/// * All pointers must be properly aligned for f32
/// * `size` must accurately reflect the size of the input tensors
/// * If provided, the CUDA stream must be valid and not dropped while operation is in progress
pub unsafe fn cuda_tensor_add(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_add(output, input1, input2, size, stream_handle);
    Ok(())
}

/// Performs element-wise subtraction of two tensors on CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires that:
/// * `output`, `input1`, and `input2` must be valid CUDA device pointers
/// * `output` must point to a buffer with size >= `size`
/// * `input1` and `input2` must point to buffers with size >= `size`
/// * All pointers must be properly aligned for f32
/// * `size` must accurately reflect the size of the input tensors
/// * If provided, the CUDA stream must be valid and not dropped while operation is in progress
pub unsafe fn cuda_tensor_sub(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_sub(output, input1, input2, size, stream_handle);
    Ok(())
}

/// Performs element-wise multiplication of two tensors on CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires that:
/// * `output`, `input1`, and `input2` must be valid CUDA device pointers
/// * `output` must point to a buffer with size >= `size`
/// * `input1` and `input2` must point to buffers with size >= `size`
/// * All pointers must be properly aligned for f32
/// * `size` must accurately reflect the size of the input tensors
/// * If provided, the CUDA stream must be valid and not dropped while operation is in progress
pub unsafe fn cuda_tensor_mul(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_mul(output, input1, input2, size, stream_handle);
    Ok(())
}

/// Performs element-wise division of two tensors on CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires that:
/// * `output`, `input1`, and `input2` must be valid CUDA device pointers
/// * `output` must point to a buffer with size >= `size`
/// * `input1` and `input2` must point to buffers with size >= `size`
/// * All pointers must be properly aligned for f32
/// * `size` must accurately reflect the size of the input tensors
/// * If provided, the CUDA stream must be valid and not dropped while operation is in progress
/// * Elements in `input2` must not be zero to avoid division by zero
pub unsafe fn cuda_tensor_div(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_div(output, input1, input2, size, stream_handle);
    Ok(())
}

/// Performs matrix multiplication of two tensors on CUDA device.
///
/// # Safety
///
/// This function is unsafe because it requires that:
/// * `output`, `input1`, and `input2` must be valid CUDA device pointers
/// * `output` must point to a buffer with size >= m * n
/// * `input1` must point to a buffer with size >= m * k
/// * `input2` must point to a buffer with size >= k * n
/// * All pointers must be properly aligned for f32
/// * Matrix dimensions (m, n, k) must be positive and accurately reflect the input tensor dimensions
/// * If provided, the CUDA stream must be valid and not dropped while operation is in progress
///
/// # Matrix Dimensions
/// * `m`: Number of rows in the first input matrix and output matrix
/// * `n`: Number of columns in the second input matrix and output matrix
/// * `k`: Number of columns in the first input matrix and rows in the second input matrix
pub unsafe fn cuda_tensor_mat_mul(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    m: i32,
    n: i32,
    k: i32,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tensor_mat_mul(output, input1, input2, m, n, k, stream_handle);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CudaBuffer, error::CudaResult};

    #[test]
    fn test_cuda_add_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input1_buf = CudaBuffer::new(size, 0)?;
        let mut input2_buf = CudaBuffer::new(size, 0)?;

        let input1_data: Vec<f32> = vec![1.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_add(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
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
    fn test_cuda_sub_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input1_buf = CudaBuffer::new(size, 0)?;
        let mut input2_buf = CudaBuffer::new(size, 0)?;

        let input1_data: Vec<f32> = vec![3.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_sub(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                1.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_cuda_mul_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input1_buf = CudaBuffer::new(size, 0)?;
        let mut input2_buf = CudaBuffer::new(size, 0)?;

        let input1_data: Vec<f32> = vec![3.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_mul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
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
    fn test_cuda_div_with_stream() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input1_buf = CudaBuffer::new(size, 0)?;
        let mut input2_buf = CudaBuffer::new(size, 0)?;

        let input1_data: Vec<f32> = vec![6.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_div(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
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
    fn test_cuda_mat_mul_with_stream() -> CudaResult<()> {
        let m = 32;
        let n = 32;
        let k = 32;
        let size = (m * n) as usize;
        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input1_buf = CudaBuffer::new(size, 0)?;
        let mut input2_buf = CudaBuffer::new(size, 0)?;

        let input1_data: Vec<f32> = vec![1.0; (m * k) as usize];
        let input2_data: Vec<f32> = vec![2.0; (k * n) as usize];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tensor_mat_mul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                m,
                n,
                k,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - (k as f32 * 2.0)).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                k as f32 * 2.0,
                val
            );
        }

        Ok(())
    }
}
