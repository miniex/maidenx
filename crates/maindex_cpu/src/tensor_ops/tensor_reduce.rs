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
}
