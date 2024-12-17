use crate::error::CpuResult;

/// Performs the forward pass of the Linear layer on the CPU.
///
/// # Safety
///
/// * `output`, `input`, `weights`, and `bias` must be valid pointers to CPU memory.
/// * The output buffer must have size >= batch_size * output_dim
/// * The input buffer must have size >= batch_size * input_dim
/// * The weights buffer must have size >= input_dim * output_dim
/// * The bias buffer must have size >= output_dim
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_linear_forward(
    output: *mut f32,
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
) -> CpuResult<()> {
    for b in 0..batch_size {
        for o in 0..output_dim {
            let mut sum = if bias.is_null() { 0.0 } else { *bias.add(o) };
            for i in 0..input_dim {
                sum += *input.add(b * input_dim + i) * *weights.add(o * input_dim + i);
            }
            *output.add(b * output_dim + o) = sum;
        }
    }
    Ok(())
}

/// Performs the backward pass of the Linear layer on the CPU.
///
/// # Safety
///
/// * All pointers must be valid pointers to CPU memory.
/// * d_output buffer must have size >= batch_size * output_dim
/// * input buffer must have size >= batch_size * input_dim
/// * weights buffer must have size >= input_dim * output_dim
/// * d_input buffer must have size >= batch_size * input_dim
/// * d_weights buffer must have size >= input_dim * output_dim
/// * d_bias buffer must have size >= output_dim
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cpu_linear_backward(
    d_output: *const f32,
    input: *const f32,
    weights: *const f32,
    d_input: *mut f32,
    d_weights: *mut f32,
    d_bias: *mut f32,
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
) -> CpuResult<()> {
    // Initialize gradients to zero
    for i in 0..(input_dim * output_dim) {
        *d_weights.add(i) = 0.0;
    }
    for i in 0..(batch_size * input_dim) {
        *d_input.add(i) = 0.0;
    }
    if !d_bias.is_null() {
        for i in 0..output_dim {
            *d_bias.add(i) = 0.0;
        }
    }

    // Compute gradients
    for b in 0..batch_size {
        for o in 0..output_dim {
            let grad = *d_output.add(b * output_dim + o);

            // Gradient for bias
            if !d_bias.is_null() {
                *d_bias.add(o) += grad;
            }

            // Gradients for weights and input
            for i in 0..input_dim {
                let input_val = *input.add(b * input_dim + i);
                *d_weights.add(o * input_dim + i) += grad * input_val;
                *d_input.add(b * input_dim + i) += grad * *weights.add(o * input_dim + i);
            }
        }
    }
    Ok(())
}

/// Performs the forward pass of the Bilinear layer on the CPU.
///
/// # Safety
///
/// * All pointers must be valid pointers to CPU memory.
/// * output buffer must have size >= batch_size * output_dim
/// * input1 buffer must have size >= batch_size * dim1
/// * input2 buffer must have size >= batch_size * dim2
/// * weights buffer must have size >= dim1 * dim2 * output_dim
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cpu_bilinear_forward(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    weights: *const f32,
    batch_size: usize,
    dim1: usize,
    dim2: usize,
    output_dim: usize,
) -> CpuResult<()> {
    for b in 0..batch_size {
        for o in 0..output_dim {
            let mut sum = 0.0;
            for i in 0..dim1 {
                let val1 = *input1.add(b * dim1 + i);
                for j in 0..dim2 {
                    let val2 = *input2.add(b * dim2 + j);
                    sum += val1 * val2 * *weights.add((o * dim1 + i) * dim2 + j);
                }
            }
            *output.add(b * output_dim + o) = sum;
        }
    }
    Ok(())
}

/// Performs the backward pass of the Bilinear layer on the CPU.
///
/// # Safety
///
/// * All pointers must be valid pointers to CPU memory.
/// * All gradient buffers must match the dimensions of their corresponding forward pass buffers
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cpu_bilinear_backward(
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
) -> CpuResult<()> {
    // Initialize gradients to zero
    for i in 0..(batch_size * dim1) {
        *d_input1.add(i) = 0.0;
    }
    for i in 0..(batch_size * dim2) {
        *d_input2.add(i) = 0.0;
    }
    for i in 0..(dim1 * dim2 * output_dim) {
        *d_weights.add(i) = 0.0;
    }

    // Compute gradients
    for b in 0..batch_size {
        for o in 0..output_dim {
            let grad = *d_output.add(b * output_dim + o);

            for i in 0..dim1 {
                for j in 0..dim2 {
                    let val1 = *input1.add(b * dim1 + i);
                    let val2 = *input2.add(b * dim2 + j);
                    let w = *weights.add((o * dim1 + i) * dim2 + j);

                    *d_input1.add(b * dim1 + i) += grad * val2 * w;
                    *d_input2.add(b * dim2 + j) += grad * val1 * w;
                    *d_weights.add((o * dim1 + i) * dim2 + j) += grad * val1 * val2;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::CpuBuffer;

    #[test]
    fn test_cpu_linear_forward() -> CpuResult<()> {
        let batch_size = 2;
        let input_dim = 3;
        let output_dim = 2;

        let input_data = vec![
            1.0f32, 2.0, 3.0, // first batch
            4.0, 5.0, 6.0, // second batch
        ];

        let weights_data = vec![
            0.1f32, 0.2, 0.3, // weights for first output
            0.4, 0.2, 0.3, // weights for second output
        ];

        let bias_data = vec![0.1f32, 0.2];

        let mut output_buf = CpuBuffer::new(batch_size * output_dim)?;
        let mut input_buf = CpuBuffer::new(batch_size * input_dim)?;
        let mut weights_buf = CpuBuffer::new(input_dim * output_dim)?;
        let mut bias_buf = CpuBuffer::new(output_dim)?;

        input_buf.copy_from_host(&input_data)?;
        weights_buf.copy_from_host(&weights_data)?;
        bias_buf.copy_from_host(&bias_data)?;

        unsafe {
            cpu_linear_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                weights_buf.as_ptr(),
                bias_buf.as_ptr(),
                batch_size,
                input_dim,
                output_dim,
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
    fn test_cpu_linear_backward() -> CpuResult<()> {
        let batch_size = 2;
        let input_dim = 3;
        let output_dim = 2;

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.2, 0.3];
        let d_output_data = vec![1.0f32, 1.0, 1.0, 1.0];

        let mut d_input_buf = CpuBuffer::new(batch_size * input_dim)?;
        let mut d_weights_buf = CpuBuffer::new(input_dim * output_dim)?;
        let mut d_bias_buf = CpuBuffer::new(output_dim)?;
        let mut input_buf = CpuBuffer::new(batch_size * input_dim)?;
        let mut weights_buf = CpuBuffer::new(input_dim * output_dim)?;
        let mut d_output_buf = CpuBuffer::new(batch_size * output_dim)?;

        input_buf.copy_from_host(&input_data)?;
        weights_buf.copy_from_host(&weights_data)?;
        d_output_buf.copy_from_host(&d_output_data)?;

        unsafe {
            cpu_linear_backward(
                d_output_buf.as_ptr(),
                input_buf.as_ptr(),
                weights_buf.as_ptr(),
                d_input_buf.as_mut_ptr(),
                d_weights_buf.as_mut_ptr(),
                d_bias_buf.as_mut_ptr(),
                batch_size,
                input_dim,
                output_dim,
            )?;
        }

        let mut d_bias_data = vec![0.0f32; output_dim];
        d_bias_buf.copy_to_host(&mut d_bias_data)?;

        assert_eq!(d_bias_data, vec![2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_cpu_bilinear_forward() -> CpuResult<()> {
        let batch_size = 2;
        let dim1 = 2;
        let dim2 = 2;
        let output_dim = 1;

        let input1_data = vec![
            1.0f32, 2.0, // first batch
            3.0, 4.0, // second batch
        ];

        let input2_data = vec![
            0.5f32, 0.6, // first batch
            0.7, 0.8, // second batch
        ];

        let weights_data = vec![
            0.1f32, 0.2, // weights for dim1=0
            0.3, 0.4, // weights for dim1=1
        ];

        let mut output_buf = CpuBuffer::new(batch_size * output_dim)?;
        let mut input1_buf = CpuBuffer::new(batch_size * dim1)?;
        let mut input2_buf = CpuBuffer::new(batch_size * dim2)?;
        let mut weights_buf = CpuBuffer::new(dim1 * dim2 * output_dim)?;

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;
        weights_buf.copy_from_host(&weights_data)?;

        unsafe {
            cpu_bilinear_forward(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                weights_buf.as_ptr(),
                batch_size,
                dim1,
                dim2,
                output_dim,
            )?;
        }

        let mut output_data = vec![0.0f32; batch_size * output_dim];
        output_buf.copy_to_host(&mut output_data)?;

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
    fn test_cpu_bilinear_backward() -> CpuResult<()> {
        let batch_size = 2;
        let dim1 = 2;
        let dim2 = 2;
        let output_dim = 1;

        let input1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input2_data = vec![0.5f32, 0.6, 0.7, 0.8];
        let weights_data = vec![0.1f32, 0.2, 0.3, 0.4];
        let d_output_data = vec![1.0f32, 1.0]; // Gradient from next layer

        let mut d_input1_buf = CpuBuffer::new(batch_size * dim1)?;
        let mut d_input2_buf = CpuBuffer::new(batch_size * dim2)?;
        let mut d_weights_buf = CpuBuffer::new(dim1 * dim2 * output_dim)?;
        let mut input1_buf = CpuBuffer::new(batch_size * dim1)?;
        let mut input2_buf = CpuBuffer::new(batch_size * dim2)?;
        let mut weights_buf = CpuBuffer::new(dim1 * dim2 * output_dim)?;
        let mut d_output_buf = CpuBuffer::new(batch_size * output_dim)?;

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;
        weights_buf.copy_from_host(&weights_data)?;
        d_output_buf.copy_from_host(&d_output_data)?;

        unsafe {
            cpu_bilinear_backward(
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
            )?;
        }

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

        // Manual verification of gradients for first batch
        let expected_d_input1_first = [
            0.5 * 0.1 + 0.6 * 0.2, // First input1 gradient
            0.5 * 0.3 + 0.6 * 0.4, // Second input1 gradient
        ];

        for (actual, expected) in d_input1_data[0..2]
            .iter()
            .zip(expected_d_input1_first.iter())
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "d_input1 mismatch: actual = {}, expected = {}",
                actual,
                expected
            );
        }

        Ok(())
    }
}
