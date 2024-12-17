use crate::error::CpuResult;

/// Performs element-wise addition of a tensor and a scalar on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Both buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_scalar_add(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input.add(i) + scalar;
    }
    Ok(())
}

/// Performs element-wise subtraction of a tensor and a scalar on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Both buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_scalar_sub(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input.add(i) - scalar;
    }
    Ok(())
}

/// Performs element-wise multiplication of a tensor and a scalar on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Both buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_scalar_mul(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input.add(i) * scalar;
    }
    Ok(())
}

/// Performs element-wise division of a tensor and a scalar on CPU.
///
/// # Safety
///
/// * `output`, `input` must be valid pointers to CPU memory
/// * Both buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
/// * Scalar must not be zero
pub unsafe fn cpu_tensor_scalar_div(
    output: *mut f32,
    input: *const f32,
    scalar: f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input.add(i) / scalar;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CpuBuffer, error::CpuResult};

    #[test]
    fn test_cpu_scalar_add() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![1.0; size];
        let scalar = 2.0;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_scalar_add(output_buf.as_mut_ptr(), input_buf.as_ptr(), scalar, size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 3.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_scalar_sub() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![3.0; size];
        let scalar = 1.0;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_scalar_sub(output_buf.as_mut_ptr(), input_buf.as_ptr(), scalar, size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 2.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_scalar_mul() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![2.0; size];
        let scalar = 3.0;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_scalar_mul(output_buf.as_mut_ptr(), input_buf.as_ptr(), scalar, size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 6.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_scalar_div() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        let input_data: Vec<f32> = vec![6.0; size];
        let scalar = 2.0;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tensor_scalar_div(output_buf.as_mut_ptr(), input_buf.as_ptr(), scalar, size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 3.0).abs() < 1e-5);
        }

        Ok(())
    }
}
