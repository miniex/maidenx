use crate::{error::CudaResult, CudaStream};
use std::ffi::c_void;

#[link(name = "nn_linear_layers")]
extern "C" {
    // Linear Layer
    fn linear_forward(
        output: *mut f32,
        input: *const f32,
        weights: *const f32,
        bias: *const f32,
        batch_size: usize,
        input_dim: usize,
        output_dim: usize,
        stream: *mut c_void,
    );
    fn linear_backward(
        d_output: *const f32,
        input: *const f32,
        weights: *const f32,
        d_input: *mut f32,
        d_weights: *mut f32,
        d_bias: *mut f32,
        batch_size: usize,
        input_dim: usize,
        output_dim: usize,
        stream: *mut c_void,
    );

    // Bilinear Layer
    fn bilinear_forward(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        weights: *const f32,
        batch_size: usize,
        dim1: usize,
        dim2: usize,
        output_dim: usize,
        stream: *mut c_void,
    );
    fn bilinear_backward(
        d_output: *const f32,
        input1: *const f32,
        input2: *const f32,
        weights: *const f32,
        d_input1: *mut f32,
        d_input2: *mut f32,
        d_weights: *mut f32,
        batch_size: usize,
        dim1: usize,
        dim2: usize,
        output_dim: usize,
        stream: *mut c_void,
    );
}

/// Performs the forward pass of a Linear layer on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `output` must be a valid CUDA device pointer with size >= `batch_size * output_dim`.
/// * `input` must be a valid CUDA device pointer with size >= `batch_size * input_dim`.
/// * `weights` must be a valid CUDA device pointer with size >= `input_dim * output_dim`.
/// * `bias` must be a valid CUDA device pointer with size >= `output_dim`.
/// * All pointers must be properly aligned for `f32`.
/// * `batch_size`, `input_dim`, and `output_dim` must accurately reflect the dimensions of the tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_linear_forward(
    output: *mut f32,
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    linear_forward(
        output,
        input,
        weights,
        bias,
        batch_size,
        input_dim,
        output_dim,
        stream_handle,
    );
    Ok(())
}

/// Performs the backward pass of a Linear layer on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `d_output` must be a valid CUDA device pointer with size >= `batch_size * output_dim`.
/// * `input` must be a valid CUDA device pointer with size >= `batch_size * input_dim`.
/// * `weights` must be a valid CUDA device pointer with size >= `input_dim * output_dim`.
/// * `d_input` must be a valid CUDA device pointer with size >= `batch_size * input_dim`.
/// * `d_weights` must be a valid CUDA device pointer with size >= `input_dim * output_dim`.
/// * `d_bias` must be a valid CUDA device pointer with size >= `output_dim`.
/// * All pointers must be properly aligned for `f32`.
/// * `batch_size`, `input_dim`, and `output_dim` must accurately reflect the dimensions of the tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_linear_backward(
    d_output: *const f32,
    input: *const f32,
    weights: *const f32,
    d_input: *mut f32,
    d_weights: *mut f32,
    d_bias: *mut f32,
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    linear_backward(
        d_output,
        input,
        weights,
        d_input,
        d_weights,
        d_bias,
        batch_size,
        input_dim,
        output_dim,
        stream_handle,
    );
    Ok(())
}

/// Performs the forward pass of a Bilinear layer on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `output` must be a valid CUDA device pointer with size >= `batch_size * output_dim`.
/// * `input1` must be a valid CUDA device pointer with size >= `batch_size * dim1`.
/// * `input2` must be a valid CUDA device pointer with size >= `batch_size * dim2`.
/// * `weights` must be a valid CUDA device pointer with size >= `dim1 * dim2 * output_dim`.
/// * All pointers must be properly aligned for `f32`.
/// * `batch_size`, `dim1`, `dim2`, and `output_dim` must accurately reflect the dimensions of the tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_bilinear_forward(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    weights: *const f32,
    batch_size: usize,
    dim1: usize,
    dim2: usize,
    output_dim: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    bilinear_forward(
        output,
        input1,
        input2,
        weights,
        batch_size,
        dim1,
        dim2,
        output_dim,
        stream_handle,
    );
    Ok(())
}

/// Performs the backward pass of a Bilinear layer on CUDA device.
///
/// # Safety
///
/// This function is unsafe because:
/// * `d_output` must be a valid CUDA device pointer with size >= `batch_size * output_dim`.
/// * `input1` must be a valid CUDA device pointer with size >= `batch_size * dim1`.
/// * `input2` must be a valid CUDA device pointer with size >= `batch_size * dim2`.
/// * `weights` must be a valid CUDA device pointer with size >= `dim1 * dim2 * output_dim`.
/// * `d_input1` must be a valid CUDA device pointer with size >= `batch_size * dim1`.
/// * `d_input2` must be a valid CUDA device pointer with size >= `batch_size * dim2`.
/// * `d_weights` must be a valid CUDA device pointer with size >= `dim1 * dim2 * output_dim`.
/// * All pointers must be properly aligned for `f32`.
/// * `batch_size`, `dim1`, `dim2`, and `output_dim` must accurately reflect the dimensions of the tensors.
/// * If provided, the CUDA stream must be valid and not dropped while the operation is in progress.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_bilinear_backward(
    d_output: *const f32,
    input1: *const f32,
    input2: *const f32,
    weights: *const f32,
    d_input1: *mut f32,
    d_input2: *mut f32,
    d_weights: *mut f32,
    batch_size: usize,
    dim1: usize,
    dim2: usize,
    output_dim: usize,
    stream: Option<&CudaStream>,
) -> CudaResult<()> {
    let stream_handle = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    bilinear_backward(
        d_output,
        input1,
        input2,
        weights,
        d_input1,
        d_input2,
        d_weights,
        batch_size,
        dim1,
        dim2,
        output_dim,
        stream_handle,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::CudaBuffer;

    #[test]
    fn test_linear_forward() -> CudaResult<()> {
        // Setup test dimensions
        let batch_size = 2;
        let input_dim = 3;
        let output_dim = 2;

        // Create test data
        let input_data = vec![
            1.0f32, 2.0, 3.0, // first batch
            4.0, 5.0, 6.0,
        ]; // second batch

        // weights: (output_dim, input_dim) format
        // [0.1, 0.2, 0.3] - first output neuron weights
        // [0.4, 0.2, 0.3] - second output neuron weights
        let weights_data = vec![
            0.1f32, 0.2, 0.3, // weights for first output
            0.4, 0.2, 0.3,
        ]; // weights for second output

        let bias_data = vec![0.1f32, 0.2]; // one bias per output

        // Create CUDA buffers
        let mut output_buf = CudaBuffer::new(batch_size * output_dim, 0)?;
        let mut input_buf = CudaBuffer::new(batch_size * input_dim, 0)?;
        let mut weights_buf = CudaBuffer::new(input_dim * output_dim, 0)?;
        let mut bias_buf = CudaBuffer::new(output_dim, 0)?;

        // Copy data to device
        input_buf.copy_from_host(&input_data)?;
        weights_buf.copy_from_host(&weights_data)?;
        bias_buf.copy_from_host(&bias_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_linear_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                weights_buf.as_ptr(),
                bias_buf.as_ptr(),
                batch_size,
                input_dim,
                output_dim,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; batch_size * output_dim];
        output_buf.copy_to_host(&mut output_data)?;

        let expected = [1.5f32, 1.9, 3.3, 4.6];

        for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Mismatch at index {}: actual = {}, expected = {}",
                i,
                actual,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn test_linear_backward() -> CudaResult<()> {
        let batch_size = 2;
        let input_dim = 3;
        let output_dim = 2;

        // Test data
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.2, 0.3];
        let d_output_data = vec![1.0f32, 1.0, 1.0, 1.0]; // Gradient from next layer

        // Create CUDA buffers
        let mut d_input_buf = CudaBuffer::new(batch_size * input_dim, 0)?;
        let mut d_weights_buf = CudaBuffer::new(input_dim * output_dim, 0)?;
        let mut d_bias_buf = CudaBuffer::new(output_dim, 0)?;
        let mut input_buf = CudaBuffer::new(batch_size * input_dim, 0)?;
        let mut weights_buf = CudaBuffer::new(input_dim * output_dim, 0)?;
        let mut d_output_buf = CudaBuffer::new(batch_size * output_dim, 0)?;

        // Copy data to device
        input_buf.copy_from_host(&input_data)?;
        weights_buf.copy_from_host(&weights_data)?;
        d_output_buf.copy_from_host(&d_output_data)?;

        let stream = CudaStream::new()?;

        // Run backward pass
        unsafe {
            cuda_linear_backward(
                d_output_buf.as_ptr(),
                input_buf.as_ptr(),
                weights_buf.as_ptr(),
                d_input_buf.as_mut_ptr(),
                d_weights_buf.as_mut_ptr(),
                d_bias_buf.as_mut_ptr(),
                batch_size,
                input_dim,
                output_dim,
                Some(&stream),
            )?;
        }

        // Get results
        let mut d_input_data = vec![0.0f32; batch_size * input_dim];
        let mut d_weights_data = vec![0.0f32; input_dim * output_dim];
        let mut d_bias_data = vec![0.0f32; output_dim];

        d_input_buf.copy_to_host(&mut d_input_data)?;
        d_weights_buf.copy_to_host(&mut d_weights_data)?;
        d_bias_buf.copy_to_host(&mut d_bias_data)?;

        // Verify gradient shapes
        assert_eq!(d_input_data.len(), batch_size * input_dim);
        assert_eq!(d_weights_data.len(), input_dim * output_dim);
        assert_eq!(d_bias_data.len(), output_dim);

        // Verify d_bias (should be sum of d_output for each output dimension)
        assert_eq!(d_bias_data, vec![2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_bilinear_forward() -> CudaResult<()> {
        let batch_size = 2;
        let dim1 = 2;
        let dim2 = 2;
        let output_dim = 1;

        // Create test data
        let input1_data = vec![
            1.0f32, 2.0, // first batch
            3.0, 4.0,
        ]; // second batch

        let input2_data = vec![
            0.5f32, 0.6, // first batch
            0.7, 0.8,
        ]; // second batch

        // weights: (output_dim, dim1, dim2) format
        let weights_data = vec![
            0.1f32, 0.2, // weights for dim1=0
            0.3, 0.4, // weights for dim1=1
        ];

        // Create CUDA buffers
        let mut output_buf = CudaBuffer::new(batch_size * output_dim, 0)?;
        let mut input1_buf = CudaBuffer::new(batch_size * dim1, 0)?;
        let mut input2_buf = CudaBuffer::new(batch_size * dim2, 0)?;
        let mut weights_buf = CudaBuffer::new(dim1 * dim2 * output_dim, 0)?;

        // Copy data to device
        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;
        weights_buf.copy_from_host(&weights_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_bilinear_forward(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                weights_buf.as_ptr(),
                batch_size,
                dim1,
                dim2,
                output_dim,
                Some(&stream),
            )?;
        }

        let mut output_data = vec![0.0f32; batch_size * output_dim];
        output_buf.copy_to_host(&mut output_data)?;

        // Manual calculation for first batch:
        // output[0] = input1[0]*input2[0]*weights[0,0] + input1[0]*input2[1]*weights[0,1] +
        //             input1[1]*input2[0]*weights[1,0] + input1[1]*input2[1]*weights[1,1]
        // = 1.0*0.5*0.1 + 1.0*0.6*0.2 + 2.0*0.5*0.3 + 2.0*0.6*0.4
        let expected_first = 1.0 * 0.5 * 0.1 + 1.0 * 0.6 * 0.2 + 2.0 * 0.5 * 0.3 + 2.0 * 0.6 * 0.4;
        let expected_second = 3.0 * 0.7 * 0.1 + 3.0 * 0.8 * 0.2 + 4.0 * 0.7 * 0.3 + 4.0 * 0.8 * 0.4;

        let expected = [expected_first, expected_second];

        for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Mismatch at index {}: actual = {}, expected = {}",
                i,
                actual,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn test_bilinear_backward() -> CudaResult<()> {
        let batch_size = 2;
        let dim1 = 2;
        let dim2 = 2;
        let output_dim = 1;

        // Test data
        let input1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input2_data = vec![0.5f32, 0.6, 0.7, 0.8];
        let weights_data = vec![0.1f32, 0.2, 0.3, 0.4];
        let d_output_data = vec![1.0f32, 1.0]; // Gradient from next layer

        // Create CUDA buffers
        let mut d_input1_buf = CudaBuffer::new(batch_size * dim1, 0)?;
        let mut d_input2_buf = CudaBuffer::new(batch_size * dim2, 0)?;
        let mut d_weights_buf = CudaBuffer::new(dim1 * dim2 * output_dim, 0)?;
        let mut input1_buf = CudaBuffer::new(batch_size * dim1, 0)?;
        let mut input2_buf = CudaBuffer::new(batch_size * dim2, 0)?;
        let mut weights_buf = CudaBuffer::new(dim1 * dim2 * output_dim, 0)?;
        let mut d_output_buf = CudaBuffer::new(batch_size * output_dim, 0)?;

        // Copy data to device
        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;
        weights_buf.copy_from_host(&weights_data)?;
        d_output_buf.copy_from_host(&d_output_data)?;

        let stream = CudaStream::new()?;

        unsafe {
            cuda_bilinear_backward(
                d_output_buf.as_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                weights_buf.as_ptr(),
                d_input1_buf.as_mut_ptr(),
                d_input2_buf.as_mut_ptr(),
                d_weights_buf.as_mut_ptr(),
                batch_size,
                dim1,
                dim2,
                output_dim,
                Some(&stream),
            )?;
        }

        // Get results
        let mut d_input1_data = vec![0.0f32; batch_size * dim1];
        let mut d_input2_data = vec![0.0f32; batch_size * dim2];
        let mut d_weights_data = vec![0.0f32; dim1 * dim2 * output_dim];

        d_input1_buf.copy_to_host(&mut d_input1_data)?;
        d_input2_buf.copy_to_host(&mut d_input2_data)?;
        d_weights_buf.copy_to_host(&mut d_weights_data)?;

        // Verify gradient shapes
        assert_eq!(d_input1_data.len(), batch_size * dim1);
        assert_eq!(d_input2_data.len(), batch_size * dim2);
        assert_eq!(d_weights_data.len(), dim1 * dim2 * output_dim);

        Ok(())
    }
}
