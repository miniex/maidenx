use crate::error::CpuResult;

/// Performs element-wise addition of two tensors on CPU.
///
/// # Safety
///
/// * `output`, `input1`, `input2` must be valid pointers to CPU memory
/// * All buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_add(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input1.add(i) + *input2.add(i);
    }
    Ok(())
}

/// Performs element-wise subtraction of two tensors on CPU.
///
/// # Safety
///
/// * `output`, `input1`, `input2` must be valid pointers to CPU memory
/// * All buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_sub(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input1.add(i) - *input2.add(i);
    }
    Ok(())
}

/// Performs element-wise multiplication of two tensors on CPU.
///
/// # Safety
///
/// * `output`, `input1`, `input2` must be valid pointers to CPU memory
/// * All buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
pub unsafe fn cpu_tensor_mul(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = *input1.add(i) * *input2.add(i);
    }
    Ok(())
}

/// Performs element-wise division of two tensors on CPU.
///
/// # Safety
///
/// * `output`, `input1`, `input2` must be valid pointers to CPU memory
/// * All buffers must have size >= `size`
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
/// * Elements in `input2` must not be zero
pub unsafe fn cpu_tensor_div(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        // Cast to f64 for higher precision during computation
        let dividend = *input1.add(i) as f64;
        let divisor = *input2.add(i) as f64;
        let result = dividend / divisor;

        // Convert back to f32 and store in output
        *output.add(i) = result as f32;
    }
    Ok(())
}

/// Performs matrix multiplication of two tensors on CPU.
///
/// # Safety
///
/// * `output`, `input1`, `input2` must be valid pointers to CPU memory
/// * `output` must have size >= m * n
/// * `input1` must have size >= m * k
/// * `input2` must have size >= k * n
/// * All pointers must be properly aligned for f32
/// * Memory regions must not overlap
/// * Matrix dimensions must be positive
pub unsafe fn cpu_tensor_mat_mul(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    m: i32,
    n: i32,
    k: i32,
) -> CpuResult<()> {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += *input1.add((i * k) as usize + p as usize)
                    * *input2.add((p * n) as usize + j as usize);
            }
            *output.add((i * n) as usize + j as usize) = sum;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CpuBuffer, error::CpuResult};

    #[test]
    fn test_cpu_add() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input1_buf = CpuBuffer::new(size)?;
        let mut input2_buf = CpuBuffer::new(size)?;

        let input1_data: Vec<f32> = vec![1.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cpu_tensor_add(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 3.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_sub() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input1_buf = CpuBuffer::new(size)?;
        let mut input2_buf = CpuBuffer::new(size)?;

        let input1_data: Vec<f32> = vec![3.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cpu_tensor_sub(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 1.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_mul() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input1_buf = CpuBuffer::new(size)?;
        let mut input2_buf = CpuBuffer::new(size)?;

        let input1_data: Vec<f32> = vec![3.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cpu_tensor_mul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 6.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_div() -> CpuResult<()> {
        let size = 1024;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input1_buf = CpuBuffer::new(size)?;
        let mut input2_buf = CpuBuffer::new(size)?;

        let input1_data: Vec<f32> = vec![6.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cpu_tensor_div(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - 3.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_cpu_mat_mul() -> CpuResult<()> {
        let m = 32;
        let n = 32;
        let k = 32;
        let size = (m * n) as usize;
        let mut output_buf = CpuBuffer::new(size)?;
        let mut input1_buf = CpuBuffer::new(size)?;
        let mut input2_buf = CpuBuffer::new(size)?;

        let input1_data: Vec<f32> = vec![1.0; (m * k) as usize];
        let input2_data: Vec<f32> = vec![2.0; (k * n) as usize];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cpu_tensor_mat_mul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                m,
                n,
                k,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for val in output_data {
            assert!((val - (k as f32 * 2.0)).abs() < 1e-5);
        }

        Ok(())
    }
}
