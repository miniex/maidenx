use crate::error::CpuResult;

/// Performs the forward pass of the ReLU activation on the CPU.
///
/// # Safety
///
/// * `output` and `input` must be valid pointers to CPU memory.
/// * The buffers pointed to by `output` and `input` must have size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_relu_forward(output: *mut f32, input: *const f32, size: usize) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = (*input.add(i)).max(0.0);
    }
    Ok(())
}

/// Performs the backward pass of the ReLU activation on the CPU.
///
/// # Safety
///
/// * `grad_input`, `grad_output`, and `input` must be valid pointers to CPU memory.
/// * The buffers pointed to by `grad_input`, `grad_output`, and `input` must have size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_relu_backward(
    grad_input: *mut f32,
    grad_output: *const f32,
    input: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *grad_input.add(i) = if *input.add(i) > 0.0 {
            *grad_output.add(i)
        } else {
            0.0
        };
    }
    Ok(())
}

/// Performs the forward pass of the Sigmoid activation on the CPU.
///
/// # Safety
///
/// * `output` and `input` must be valid pointers to CPU memory.
/// * The buffers pointed to by `output` and `input` must have size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_sigmoid_forward(
    output: *mut f32,
    input: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = 1.0 / (1.0 + (-*input.add(i)).exp());
    }
    Ok(())
}

/// Performs the backward pass of the Sigmoid activation on the CPU.
///
/// # Safety
///
/// * `grad_input`, `grad_output`, and `output` must be valid pointers to CPU memory.
/// * The buffers pointed to by `grad_input`, `grad_output`, and `output` must have size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_sigmoid_backward(
    grad_input: *mut f32,
    grad_output: *const f32,
    output: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        let sigmoid_out = *output.add(i);
        *grad_input.add(i) = *grad_output.add(i) * sigmoid_out * (1.0 - sigmoid_out);
    }
    Ok(())
}

/// Performs the forward pass of the Tanh activation on the CPU.
///
/// # Safety
///
/// * `output` and `input` must be valid pointers to CPU memory.
/// * The buffers pointed to by `output` and `input` must have size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_tanh_forward(output: *mut f32, input: *const f32, size: usize) -> CpuResult<()> {
    for i in 0..size {
        *output.add(i) = (*input.add(i)).tanh();
    }
    Ok(())
}

/// Performs the backward pass of the Tanh activation on the CPU.
///
/// # Safety
///
/// * `grad_input`, `grad_output`, and `output` must be valid pointers to CPU memory.
/// * The buffers pointed to by `grad_input`, `grad_output`, and `output` must have size >= `size`.
/// * All pointers must be properly aligned for `f32`.
/// * Memory regions must not overlap.
pub unsafe fn cpu_tanh_backward(
    grad_input: *mut f32,
    grad_output: *const f32,
    output: *const f32,
    size: usize,
) -> CpuResult<()> {
    for i in 0..size {
        let tanh_out = *output.add(i);
        *grad_input.add(i) = *grad_output.add(i) * (1.0 - tanh_out * tanh_out);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{buffer::CpuBuffer, error::CpuResult};

    #[test]
    fn test_cpu_relu_forward() -> CpuResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let size = input_data.len();

        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_relu_forward(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        assert_eq!(output_data, vec![0.0, 0.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_cpu_relu_backward() -> CpuResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let size = input_data.len();

        let mut grad_input_buf = CpuBuffer::new(size)?;
        let mut grad_output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        grad_output_buf.copy_from_host(&grad_output_data)?;
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_relu_backward(
                grad_input_buf.as_mut_ptr(),
                grad_output_buf.as_ptr(),
                input_buf.as_ptr(),
                size,
            )?;
        }

        let mut grad_input_data = vec![0.0f32; size];
        grad_input_buf.copy_to_host(&mut grad_input_data)?;

        assert_eq!(grad_input_data, vec![0.0, 0.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_cpu_sigmoid_forward() -> CpuResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let size = input_data.len();

        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_sigmoid_forward(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        assert!((output_data[0] - 0.26894).abs() < 1e-5);
        assert!((output_data[1] - 0.5).abs() < 1e-5);
        assert!((output_data[2] - 0.73105).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_cpu_sigmoid_backward() -> CpuResult<()> {
        let output_data: Vec<f32> = vec![0.26894, 0.5, 0.73105];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let size = output_data.len();

        let mut grad_input_buf = CpuBuffer::new(size)?;
        let mut grad_output_buf = CpuBuffer::new(size)?;
        let mut output_buf = CpuBuffer::new(size)?;

        grad_output_buf.copy_from_host(&grad_output_data)?;
        output_buf.copy_from_host(&output_data)?;

        unsafe {
            cpu_sigmoid_backward(
                grad_input_buf.as_mut_ptr(),
                grad_output_buf.as_ptr(),
                output_buf.as_ptr(),
                size,
            )?;
        }

        let mut grad_input_data = vec![0.0f32; size];
        grad_input_buf.copy_to_host(&mut grad_input_data)?;

        assert!((grad_input_data[0] - 0.19661).abs() < 1e-5);
        assert!((grad_input_data[1] - 0.25).abs() < 1e-5);
        assert!((grad_input_data[2] - 0.19661).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_cpu_tanh_forward() -> CpuResult<()> {
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
        let size = input_data.len();

        let mut output_buf = CpuBuffer::new(size)?;
        let mut input_buf = CpuBuffer::new(size)?;

        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cpu_tanh_forward(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        assert!((output_data[0] + 0.76159).abs() < 1e-5);
        assert!((output_data[1] - 0.0).abs() < 1e-5);
        assert!((output_data[2] - 0.76159).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_cpu_tanh_backward() -> CpuResult<()> {
        let output_data: Vec<f32> = vec![-0.76159, 0.0, 0.76159];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let size = output_data.len();

        let mut grad_input_buf = CpuBuffer::new(size)?;
        let mut grad_output_buf = CpuBuffer::new(size)?;
        let mut output_buf = CpuBuffer::new(size)?;

        grad_output_buf.copy_from_host(&grad_output_data)?;
        output_buf.copy_from_host(&output_data)?;

        unsafe {
            cpu_tanh_backward(
                grad_input_buf.as_mut_ptr(),
                grad_output_buf.as_ptr(),
                output_buf.as_ptr(),
                size,
            )?;
        }

        let mut grad_input_data = vec![0.0f32; size];
        grad_input_buf.copy_to_host(&mut grad_input_data)?;

        assert!((grad_input_data[0] - 0.41989).abs() < 1e-4);
        assert!((grad_input_data[1] - 1.0).abs() < 1e-4);
        assert!((grad_input_data[2] - 0.41989).abs() < 1e-4);

        Ok(())
    }
}
