#![allow(clippy::excessive_precision)]

use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for test data
const TEST_DATA_F32: [f32; 4] = [-1.0f32, 0.0, 2.0, -3.0];
const TEST_DATA_U32: [u32; 3] = [1, 2, 0];

// Helper functions
fn setup_tensor<T: Clone + 'static>(data: Vec<T>, device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = Tensor::new(data)?;
    tensor.with_device(device)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor(data, device, dtype)?;
    tensor.with_grad().ok();

    Ok(tensor)
}

mod test_functions {
    use super::*;

    pub fn neg_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.neg()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, -2.0, 3.0]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.0; 4]);
        }

        Ok(())
    }

    pub fn abs_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.abs()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 2.0, 3.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.0, 0.0, 1.0, -1.0]);
        }

        Ok(())
    }

    pub fn sign_test(device: Device, dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U32 => {
                let x = setup_grad_tensor(TEST_DATA_U32.to_vec(), device, dtype)?;

                let result = x.sign()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 0.0]);
                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0; 3]);
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

                let result = x.sign()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-1.0, 0.0, 1.0, -1.0]);
                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0; 4]);
                }
            }
        }
        Ok(())
    }

    pub fn square_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_U32.to_vec(), device, dtype)?;

        let result = x.square()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 4.0, 0.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![2.0, 4.0, 0.0]);
        }

        Ok(())
    }

    pub fn sqrt_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_U32.to_vec(), device, dtype)?;
        let result = x.sqrt()?;
        result.backward()?;

        match dtype {
            DType::BF16 | DType::F16 => {
                let expected_grad = vec![0.5, 0.35351563, f32::INFINITY];
                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, expected_grad);
                }
            }
            _ => {
                let expected_grad = vec![0.5, 0.35355338, f32::INFINITY];
                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, expected_grad);
                }
            }
        }

        Ok(())
    }

    pub fn relu_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.relu()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 2.0, 0.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 1.0, 0.0]);
        }

        Ok(())
    }

    pub fn sigmoid_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.sigmoid()?;
        result.backward()?;

        let expected_output = [
            1.0 / (1.0 + (-(-1.0f32)).exp()),
            0.5,
            1.0 / (1.0 + (-2.0f32).exp()),
            1.0 / (1.0 + (-(-3.0f32)).exp()),
        ];

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let sigmoid_x = 1.0 / (1.0 + (-x_val).exp());
                let expected_grad = sigmoid_x * (1.0 - sigmoid_x);
                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-5,
                };
                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn tanh_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.tanh()?;
        result.backward()?;

        let expected_output = [(-1.0f32).tanh(), 0.0f32.tanh(), 2.0f32.tanh(), (-3.0f32).tanh()];

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let tanh_x = x_val.tanh();
                let expected_grad = 1.0 - tanh_x * tanh_x;
                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-5,
                };
                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn gelu_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.gelu()?;
        result.backward()?;

        let sqrt_2_over_pi = 0.7978845608028654;
        let coeff = 0.044715;

        let mut expected_output = Vec::new();
        for &val in &TEST_DATA_F32 {
            let tanh_arg = sqrt_2_over_pi * (val + coeff * val.powi(3));
            let gelu_val = 0.5 * val * (1.0 + tanh_arg.tanh());
            expected_output.push(gelu_val);
        }

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let tanh_arg = sqrt_2_over_pi * (x_val + coeff * x_val.powi(3));
                let tanh_val = tanh_arg.tanh();

                let sech_squared = 1.0 - tanh_val * tanh_val;
                let inner_derivative = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_val.powi(2));

                let expected_grad = 0.5 * (1.0 + tanh_val) + 0.5 * x_val * sech_squared * inner_derivative;

                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-4,
                };

                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn sin_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.sin()?;
        result.backward()?;

        let expected_output = [(-1.0f32).sin(), 0.0f32.sin(), 2.0f32.sin(), (-3.0f32).sin()];

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let expected_grad = x_val.cos();
                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-5,
                };
                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn cos_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.cos()?;
        result.backward()?;

        let expected_output = [(-1.0f32).cos(), 0.0f32.cos(), 2.0f32.cos(), (-3.0f32).cos()];

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let expected_grad = -x_val.sin();
                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-5,
                };
                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn tan_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;

        let result = x.tan()?;
        result.backward()?;

        let expected_output = [(-1.0f32).tan(), 0.0f32.tan(), 2.0f32.tan(), (-3.0f32).tan()];

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let cos_x = x_val.cos();
                let expected_grad = 1.0 / (cos_x * cos_x);
                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 5e-2,
                    _ => 1e-4,
                };
                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance || (grad_vec[i].is_infinite() && expected_grad.is_infinite()),
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn logical_not_test(device: Device, dtype: DType) -> Result<()> {
        let test_data = vec![1, 0, 1, 0];
        let x = setup_tensor(test_data, device, dtype)?;

        let result = x.logical_not()?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, false, true]);
        Ok(())
    }

    pub fn add_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 2.0;

        let result = x.add_scalar(scalar)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 4.0, -1.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0; 4]);
        }

        Ok(())
    }

    pub fn sub_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 2.0;

        let result = x.sub_scalar(scalar)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![-3.0, -2.0, 0.0, -5.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0; 4]);
        }

        Ok(())
    }

    pub fn mul_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 2.0;

        let result = x.mul_scalar(scalar)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![-2.0, 0.0, 4.0, -6.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![2.0; 4]);
        }

        Ok(())
    }

    pub fn div_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 2.0;

        let result = x.div_scalar(scalar)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![-0.5, 0.0, 1.0, -1.5]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5; 4]);
        }

        Ok(())
    }

    pub fn pow_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_U32.to_vec(), device, dtype)?;
        let exponent = 2.0;

        let result = x.pow(exponent)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 4.0, 0.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![2.0, 4.0, 0.0]);
        }

        Ok(())
    }

    pub fn leaky_relu_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let alpha = 0.1;

        let result = x.leaky_relu(alpha)?;
        result.backward()?;

        let expected_output = [-0.1, 0.0, 2.0, -0.3];
        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let expected_grad = [0.1, 0.1, 1.0, 0.1];
            let grad_vec = g.to_flatten_vec::<f32>()?;

            for (i, (actual, expected)) in grad_vec.iter().zip(expected_grad.iter()).enumerate() {
                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-5,
                };
                assert!(
                    (actual - expected).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    actual,
                    expected,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn elu_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let alpha = 1.0;

        let result = x.elu(alpha)?;
        result.backward()?;

        let mut expected_output = Vec::new();
        for &val in &TEST_DATA_F32 {
            if val > 0.0 {
                expected_output.push(val);
            } else {
                expected_output.push(alpha * (val.exp() - 1.0));
            }
        }

        let result_vec = result.to_flatten_vec::<f32>()?;
        for (a, b) in result_vec.iter().zip(expected_output.iter()) {
            let tolerance = match dtype {
                DType::BF16 | DType::F16 => 1e-2,
                _ => 1e-5,
            };
            assert!((a - b).abs() < tolerance, "Values differ: {} vs {} (tolerance: {})", a, b, tolerance);
        }

        if let Some(g) = x.grad()? {
            let grad_vec = g.to_flatten_vec::<f32>()?;
            for (i, &x_val) in TEST_DATA_F32.iter().enumerate() {
                let expected_grad = if x_val > 0.0 { 1.0 } else { alpha * x_val.exp() };

                let tolerance = match dtype {
                    DType::BF16 | DType::F16 => 1e-2,
                    _ => 1e-5,
                };

                assert!(
                    (grad_vec[i] - expected_grad).abs() < tolerance,
                    "Gradients differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    grad_vec[i],
                    expected_grad,
                    tolerance
                );
            }
        }

        Ok(())
    }

    pub fn eq_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 0.0;

        let result = x.eq_scalar(scalar)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, false, false]);
        Ok(())
    }

    pub fn ne_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 0.0;

        let result = x.ne_scalar(scalar)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, true, true]);
        Ok(())
    }

    pub fn lt_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 0.0;

        let result = x.lt_scalar(scalar)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, false, true]);
        Ok(())
    }

    pub fn le_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 0.0;

        let result = x.le_scalar(scalar)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, false, true]);
        Ok(())
    }

    pub fn gt_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 0.0;

        let result = x.gt_scalar(scalar)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, false, true, false]);
        Ok(())
    }

    pub fn ge_scalar_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let scalar = 0.0;

        let result = x.ge_scalar(scalar)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
        Ok(())
    }
}

// neg operation tests
mod neg {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::F64)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::neg_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::neg_test(Device::CUDA(0), DType::I64)
        }
    }
}

// abs operation tests
mod abs {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::F64)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::abs_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::abs_test(Device::CUDA(0), DType::I64)
        }
    }
}

// sign operation tests
mod sign {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sign_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sign_test(Device::CUDA(0), DType::I64)
        }
    }
}

// square operation tests
mod square {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::square_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::square_test(Device::CUDA(0), DType::I64)
        }
    }
}

// sqrt operation tests
mod sqrt {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sqrt_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sqrt_test(Device::CUDA(0), DType::I64)
        }
    }
}

// relu operation tests
mod relu {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::relu_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::relu_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::relu_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::relu_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::relu_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::relu_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::relu_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::relu_test(Device::CUDA(0), DType::F64)
        }
    }
}

// sigmoid operation tests
mod sigmoid {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sigmoid_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sigmoid_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sigmoid_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sigmoid_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sigmoid_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sigmoid_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sigmoid_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sigmoid_test(Device::CUDA(0), DType::F64)
        }
    }
}

// tanh operation tests
mod tanh {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::tanh_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::tanh_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::tanh_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::tanh_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::tanh_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::tanh_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::tanh_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::tanh_test(Device::CUDA(0), DType::F64)
        }
    }
}

// gelu operation tests
mod gelu {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::gelu_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::gelu_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::gelu_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::gelu_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::gelu_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::gelu_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::gelu_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::gelu_test(Device::CUDA(0), DType::F64)
        }
    }
}

// sin operation tests
mod sin {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sin_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sin_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sin_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sin_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sin_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sin_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sin_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sin_test(Device::CUDA(0), DType::F64)
        }
    }
}

// cos operation tests
mod cos {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::cos_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::cos_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::cos_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::cos_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::cos_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::cos_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::cos_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::cos_test(Device::CUDA(0), DType::F64)
        }
    }
}

// tan operation tests
mod tan {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::tan_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::tan_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::tan_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::tan_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::tan_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::tan_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::tan_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::tan_test(Device::CUDA(0), DType::F64)
        }
    }
}

// logical_not operation tests
mod logical_not {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_not_test(Device::CPU, DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_not_test(Device::CPU, DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_not_test(Device::CPU, DType::I32)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_not_test(Device::CUDA(0), DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_not_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_not_test(Device::CUDA(0), DType::I32)
        }
    }
}

// add_scalar operation tests
mod add_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::add_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::add_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::add_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::add_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::add_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::add_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::add_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::add_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// sub_scalar operation tests
mod sub_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sub_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sub_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sub_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sub_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sub_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sub_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sub_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sub_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// mul_scalar operation tests
mod mul_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mul_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mul_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mul_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mul_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mul_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mul_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mul_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mul_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// div_scalar operation tests
mod div_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::div_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::div_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::div_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::div_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::div_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::div_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::div_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::div_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// pow operation tests
mod pow {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::pow_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::pow_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::pow_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::pow_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::pow_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::pow_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::pow_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::pow_test(Device::CUDA(0), DType::F64)
        }
    }
}

// leaky_relu operation tests
mod leaky_relu {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::leaky_relu_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::leaky_relu_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::leaky_relu_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::leaky_relu_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::leaky_relu_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::leaky_relu_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::leaky_relu_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::leaky_relu_test(Device::CUDA(0), DType::F64)
        }
    }
}

// elu operation tests
mod elu {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::elu_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::elu_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::elu_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::elu_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::elu_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::elu_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::elu_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::elu_test(Device::CUDA(0), DType::F64)
        }
    }
}

// eq_scalar operation tests
mod eq_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::eq_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::eq_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::eq_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::eq_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::eq_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::eq_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::eq_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::eq_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// ne_scalar operation tests
mod ne_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ne_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ne_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ne_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ne_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ne_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ne_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ne_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ne_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// lt_scalar operation tests
mod lt_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::lt_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::lt_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::lt_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::lt_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::lt_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::lt_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::lt_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::lt_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// le_scalar operation tests
mod le_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::le_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::le_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::le_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::le_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::le_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::le_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::le_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::le_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// gt_scalar operation tests
mod gt_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::gt_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::gt_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::gt_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::gt_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::gt_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::gt_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::gt_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::gt_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}

// ge_scalar operation tests
mod ge_scalar {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ge_scalar_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ge_scalar_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ge_scalar_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ge_scalar_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ge_scalar_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ge_scalar_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ge_scalar_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ge_scalar_test(Device::CUDA(0), DType::F64)
        }
    }
}
