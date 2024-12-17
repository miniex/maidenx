use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "nn_non_linear_activations")]
extern "C" {
    // ReLU
    fn relu_forward(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
    fn relu_backward(
        grad_input: *mut f32,
        grad_output: *const f32,
        input: *const f32,
        size: usize,
        stream: *mut c_void,
    );

    // Sigmoid
    fn sigmoid_forward(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
    fn sigmoid_backward(
        grad_input: *mut f32,
        grad_output: *const f32,
        output: *const f32,
        size: usize,
        stream: *mut c_void,
    );

    // Tanh
    fn tanh_forward(output: *mut f32, input: *const f32, size: usize, stream: *mut c_void);
    fn tanh_backward(
        grad_input: *mut f32,
        grad_output: *const f32,
        output: *const f32,
        size: usize,
        stream: *mut c_void,
    );
}

/// Performs the forward pass of ReLU activation on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `output` and `input` must be valid CUDA device pointers.
/// * `output` must point to a buffer with size >= `size`.
/// * `input` must point to a buffer with size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * `size` must accurately reflect the size of the input tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
pub unsafe fn cuda_relu_forward(
    output: *mut f32,
    input: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    relu_forward(output, input, size, stream_handle);
    Ok(())
}

/// Performs the backward pass of ReLU activation on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `grad_input`, `grad_output`, and `input` must be valid CUDA device pointers.
/// * `grad_input` and `grad_output` must point to buffers with size >= `size`.
/// * `input` must point to a buffer with size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * `size` must accurately reflect the size of the input tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
pub unsafe fn cuda_relu_backward(
    grad_input: *mut f32,
    grad_output: *const f32,
    input: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    relu_backward(grad_input, grad_output, input, size, stream_handle);
    Ok(())
}

/// Performs the forward pass of Sigmoid activation on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `output` and `input` must be valid CUDA device pointers.
/// * `output` must point to a buffer with size >= `size`.
/// * `input` must point to a buffer with size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * `size` must accurately reflect the size of the input tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
pub unsafe fn cuda_sigmoid_forward(
    output: *mut f32,
    input: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    sigmoid_forward(output, input, size, stream_handle);
    Ok(())
}

/// Performs the backward pass of Sigmoid activation on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `grad_input`, `grad_output`, and `output` must be valid CUDA device pointers.
/// * `grad_input` and `grad_output` must point to buffers with size >= `size`.
/// * `output` must point to a buffer with size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * `size` must accurately reflect the size of the input tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
pub unsafe fn cuda_sigmoid_backward(
    grad_input: *mut f32,
    grad_output: *const f32,
    output: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    sigmoid_backward(grad_input, grad_output, output, size, stream_handle);
    Ok(())
}

/// Performs the forward pass of Tanh activation on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `output` and `input` must be valid CUDA device pointers.
/// * `output` must point to a buffer with size >= `size`.
/// * `input` must point to a buffer with size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * `size` must accurately reflect the size of the input tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
pub unsafe fn cuda_tanh_forward(
    output: *mut f32,
    input: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tanh_forward(output, input, size, stream_handle);
    Ok(())
}

/// Performs the backward pass of Tanh activation on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `grad_input`, `grad_output`, and `output` must be valid CUDA device pointers.
/// * `grad_input` and `grad_output` must point to buffers with size >= `size`.
/// * `output` must point to a buffer with size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * `size` must accurately reflect the size of the input tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
pub unsafe fn cuda_tanh_backward(
    grad_input: *mut f32,
    grad_output: *const f32,
    output: *const f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    tanh_backward(grad_input, grad_output, output, size, stream_handle);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CudaBuffer, error::CudaResult};

    #[test]
    fn test_relu_forward() -> CudaResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let size = input_data.len();

        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_relu_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        assert_eq!(output_data[0], 0.0);
        assert_eq!(output_data[1], 0.0);
        assert_eq!(output_data[2], 1.0);

        Ok(())
    }

    #[test]
    fn test_relu_backward() -> CudaResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let grad_output_data: Vec<f32> = vec![1.0; input_data.len()];
        let size = input_data.len();

        let mut grad_input_buf = CudaBuffer::new(size, 0)?;
        let mut grad_output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;

        grad_output_buf.copy_from_host(&grad_output_data)?;
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_relu_backward(
                grad_input_buf.as_mut_ptr(),
                grad_output_buf.as_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut grad_input_data = vec![0.0f32; size];
        grad_input_buf.copy_to_host(&mut grad_input_data)?;

        assert_eq!(grad_input_data[0], 0.0);
        assert_eq!(grad_input_data[1], 0.0);
        assert_eq!(grad_input_data[2], 1.0);

        Ok(())
    }

    #[test]
    fn test_sigmoid_forward() -> CudaResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let size = input_data.len();

        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_sigmoid_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        assert!((output_data[0] - 0.26894).abs() < 1e-5);
        assert!((output_data[1] - 0.5).abs() < 1e-5);
        assert!((output_data[2] - 0.73105).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_tanh_forward() -> CudaResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let size = input_data.len();

        let mut output_buf = CudaBuffer::new(size, 0)?;
        let mut input_buf = CudaBuffer::new(size, 0)?;
        input_buf.copy_from_host(&input_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_tanh_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                size,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        assert!((output_data[0] + 0.76159).abs() < 1e-5);
        assert!((output_data[1] - 0.0).abs() < 1e-5);
        assert!((output_data[2] - 0.76159).abs() < 1e-5);

        Ok(())
    }
}
