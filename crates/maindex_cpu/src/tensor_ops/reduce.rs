use crate::error::CpuResult;

/// Calculates the mean of all elements in a tensor on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Output buffer must have size >= 1
/// * Input buffer must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_mean(output: *mut f32, input: *const f32, size: usize) -> CpuResult<()> {
    let mut sum = 0.0;
    for i in 0..size {
        sum += *input.add(i);
    }
    *output = sum / (size as f32);
    Ok(())
}

/// Calculates the sum of all elements in a tensor on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Output buffer must have size >= 1
/// * Input buffer must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_sum(output: *mut f32, input: *const f32, size: usize) -> CpuResult<()> {
    let mut sum = 0.0;
    for i in 0..size {
        sum += *input.add(i);
    }
    *output = sum;
    Ok(())
}

/// Calculates the sum along a specified dimension in a tensor on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Output buffer must have sufficient size to hold the result
/// * Input buffer must have sufficient size based on input_shape
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
/// * input_shape must contain valid dimensions
/// * reduction_dim must be less than input_shape.len()
pub unsafe fn cpu_tensor_sum_with_dim(
    output: *mut f32,
    input: *const f32,
    input_shape: &[i32],
    reduction_dim: i32,
) -> CpuResult<()> {
    let ndim = input_shape.len();
    let reduction_dim = reduction_dim as usize;

    // Calculate output shape and size
    let output_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != reduction_dim)
        .map(|(_, &dim)| dim as usize)
        .collect();

    // Calculate strides for both input and output
    let mut input_strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1] as usize;
    }

    let mut output_strides = vec![1; output_shape.len()];
    for i in (0..output_shape.len() - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    let output_size = output_shape.iter().product();

    // Process each output position
    for out_idx in 0..output_size {
        // Convert output index to output coordinates
        let mut out_coords = vec![0; output_shape.len()];
        let mut remaining = out_idx;

        for (dim, stride) in output_strides.iter().enumerate() {
            out_coords[dim] = remaining / stride;
            remaining %= stride;
        }

        // Map output coordinates to input coordinates
        let mut in_coords = vec![0; ndim];
        let mut out_pos = 0;

        for (dim, coord) in in_coords.iter_mut().enumerate() {
            if dim != reduction_dim {
                *coord = out_coords[out_pos];
                out_pos += 1;
            }
        }

        // Sum along reduction dimension
        let mut sum = 0.0;
        for red_idx in 0..input_shape[reduction_dim] as usize {
            in_coords[reduction_dim] = red_idx;

            // Calculate input index using input strides
            let input_idx = in_coords
                .iter()
                .zip(input_strides.iter())
                .map(|(&coord, &stride)| coord * stride)
                .sum();

            sum += *input.add(input_idx);
        }

        *output.add(out_idx) = sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CpuBuffer, error::CpuResult};

    #[test]
    fn test_cpu_mean() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(1)?; // Single output for mean
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![2.0; size];
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_mean(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
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
    fn test_cpu_sum() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(1)?; // Single output for sum
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![2.0; size];
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_sum(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
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
    fn test_cpu_mean_zero_elements() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(1)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![0.0; size];
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_mean(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
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
    fn test_cpu_sum_zero_elements() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(1)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![0.0; size];
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_sum(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
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
    fn test_cpu_mean_various_values() -> CpuResult<()> {
        let size = 4;
        let mut output_buf = CpuBuffer::new(1)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_mean(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        assert!(
            (output_data[0] - 2.5).abs() < 1e-5,
            "Mean mismatch: expected {}, got {}",
            2.5,
            output_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_sum_various_values() -> CpuResult<()> {
        let size = 4;
        let mut output_buf = CpuBuffer::new(1)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_sum(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        assert!(
            (output_data[0] - 10.0).abs() < 1e-5,
            "Sum mismatch: expected {}, got {}",
            10.0,
            output_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_sum_with_dim_2d() -> CpuResult<()> {
        let input_shape = vec![2, 3]; // 2x3 matrix
        let size = input_shape.iter().product::<i32>() as usize;
        let mut input_buf = CpuBuffer::new(size)?;

        // Input:
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        input_buf.copy_from_host(&input_data)?;

        // Test reduction along dimension 1 (columns)
        let reduction_dim = 1;
        let output_size = 2; // 2 rows after reduction
        let mut output_buf = CpuBuffer::new(output_size)?;

        unsafe {
            cpu_tensor_sum_with_dim(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                &input_shape,
                reduction_dim,
            )?;
        }

        let mut output_data = vec![0.0f32; output_size];
        output_buf.copy_to_host(&mut output_data)?;

        // Expected sums: [6.0, 15.0]
        assert!(
            (output_data[0] - 6.0).abs() < 1e-5 && (output_data[1] - 15.0).abs() < 1e-5,
            "Sum along columns mismatch: expected [6.0, 15.0], got [{}, {}]",
            output_data[0],
            output_data[1]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_sum_with_dim_3d() -> CpuResult<()> {
        let input_shape = vec![2, 3, 2]; // 2x3x2 tensor
        let size = input_shape.iter().product::<i32>() as usize;
        let mut input_buf = CpuBuffer::new(size)?;

        // Input tensor (2,3,2):
        // [[[0, 1],  [2, 3],  [4, 5]],
        //  [[6, 7],  [8, 9], [10,11]]]
        let input_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        // Test reduction along dimension 1 (middle dimension)
        let reduction_dim = 1;
        let output_size = 4; // 2x2 after reduction
        let mut output_buf = CpuBuffer::new(output_size)?;

        unsafe {
            cpu_tensor_sum_with_dim(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                &input_shape,
                reduction_dim,
            )?;
        }

        let mut output_data = vec![0.0f32; output_size];
        output_buf.copy_to_host(&mut output_data)?;

        // Expected results after summing middle dimension (dim=1)
        // For first output row (sum of [[[0,1],[2,3],[4,5]])
        // [6,9]  where 6 = 0+2+4 and 9 = 1+3+5
        // For second output row (sum of [[[6,7],[8,9],[10,11]])
        // [24,27] where 24 = 6+8+10 and 27 = 7+9+11
        let expected = [6.0, 9.0, 24.0, 27.0];

        for (i, (&actual, &expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "3D sum mismatch at position {}: got {}, expected {}",
                i,
                actual,
                expected
            );
        }

        Ok(())
    }
}
