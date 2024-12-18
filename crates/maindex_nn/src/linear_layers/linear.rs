use crate::{error::NnResult, module::Module};
use maidenx_cpu::nn_layers::linear::{cpu_linear_backward, cpu_linear_forward};
#[cfg(feature = "cuda")]
use maidenx_cuda::nn_layers::linear::{cuda_linear_backward, cuda_linear_forward};
use maidenx_device::{get_current_device, Device};
use maidenx_tensor::{gradient::Node, tape::TENSOR_TAPE, Tensor};
use std::{
    ptr,
    sync::{Arc, Mutex},
};

#[derive(Module)]
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    device: Device,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        weight: Option<Tensor>,
        bias: bool,
    ) -> NnResult<Self> {
        let device = get_current_device();
        Self::new_with_device(in_features, out_features, weight, bias, device)
    }

    pub fn new_with_device(
        in_features: usize,
        out_features: usize,
        weight: Option<Tensor>,
        bias: bool,
        device: Device,
    ) -> NnResult<Self> {
        let mut weight = weight.unwrap_or_else(|| {
            let k = 1.0 / (in_features as f32).sqrt();
            Tensor::randn_with_device(&[out_features, in_features], &device)
                .unwrap()
                .scalar_mul(k)
                .unwrap()
        });
        weight.with_grad();

        let bias = if bias {
            let mut b = Tensor::zeros_with_device(&[out_features], &device).unwrap();
            b.with_grad();
            Some(b)
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
            device,
        })
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let batch_size = input.shape()[0];
        let output_shape = &[batch_size, self.out_features];
        let mut output = Tensor::from_vec_with_device(
            vec![0.0; output_shape.iter().product()],
            output_shape,
            &self.device,
        )?;

        let bias_ptr = self
            .bias
            .as_ref()
            .map_or(ptr::null(), |b| b.buffer().as_ptr());

        unsafe {
            match self.device {
                Device::Cpu => {
                    cpu_linear_forward(
                        output.buffer_mut().as_mut_ptr(),
                        input.buffer().as_ptr(),
                        self.weight.buffer().as_ptr(),
                        bias_ptr,
                        batch_size,
                        self.in_features,
                        self.out_features,
                    )?;
                }
                #[cfg(feature = "cuda")]
                Device::Cuda(_) => {
                    cuda_linear_forward(
                        output.buffer_mut().as_mut_ptr(),
                        input.buffer().as_ptr(),
                        self.weight.buffer().as_ptr(),
                        bias_ptr,
                        batch_size,
                        self.in_features,
                        self.out_features,
                        None,
                    )?;
                }
            }
        }

        let node = if input.is_requires_grad() {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let input = input.clone();
                    let weight = self.weight.clone();
                    let bias = self.bias.clone();

                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut d_input =
                            Tensor::zeros_with_device(input.shape(), input.device()).unwrap();
                        let mut d_weights =
                            Tensor::zeros_with_device(weight.shape(), weight.device()).unwrap();
                        let mut d_bias = bias
                            .as_ref()
                            .map(|b| Tensor::zeros_with_device(b.shape(), b.device()).unwrap());

                        unsafe {
                            match input.device() {
                                Device::Cpu => {
                                    cpu_linear_backward(
                                        grad_output.buffer().as_ptr(),
                                        input.buffer().as_ptr(),
                                        weight.buffer().as_ptr(),
                                        d_input.buffer_mut().as_mut_ptr(),
                                        d_weights.buffer_mut().as_mut_ptr(),
                                        d_bias.as_mut().map_or(ptr::null_mut(), |b| {
                                            b.buffer_mut().as_mut_ptr()
                                        }),
                                        input.shape()[0],
                                        weight.shape()[1],
                                        weight.shape()[0],
                                    )
                                    .unwrap();
                                }
                                #[cfg(feature = "cuda")]
                                Device::Cuda(_) => {
                                    cuda_linear_backward(
                                        grad_output.buffer().as_ptr(),
                                        input.buffer().as_ptr(),
                                        weight.buffer().as_ptr(),
                                        d_input.buffer_mut().as_mut_ptr(),
                                        d_weights.buffer_mut().as_mut_ptr(),
                                        d_bias.as_mut().map_or(ptr::null_mut(), |b| {
                                            b.buffer_mut().as_mut_ptr()
                                        }),
                                        input.shape()[0],
                                        weight.shape()[1],
                                        weight.shape()[0],
                                        None,
                                    )
                                    .unwrap();
                                }
                            }
                        }

                        let mut grads = vec![d_input, d_weights];
                        if let Some(bias_grad) = d_bias {
                            grads.push(bias_grad);
                        }
                        grads
                    }
                })),
                inputs: vec![
                    input.node().map(|n| Arc::downgrade(&n)).unwrap_or_default(),
                    self.weight
                        .node()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    self.bias
                        .as_ref()
                        .and_then(|b| b.node().map(|n| Arc::downgrade(&n)))
                        .unwrap_or_default(),
                ],
                grad: None,
            })))
        } else {
            None
        };

        output.set_requires_grad(input.is_requires_grad());
        output.set_node(node);

        TENSOR_TAPE.with(|tape| tape.borrow_mut().add(output.clone()));

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_device::Device;

    #[test]
    fn test_linear_forward_with_bias() -> NnResult<()> {
        let device = Device::cpu();
        let input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        let linear = Linear::new_with_device(3, 2, None, true, device)?;
        let output = linear.forward(&input)?;

        println!("Output with bias: {:?}", output.to_vec()?);
        Ok(())
    }

    #[test]
    fn test_linear_forward_without_bias() -> NnResult<()> {
        let device = Device::cpu();
        let input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        let linear = Linear::new_with_device(3, 2, None, false, device)?;
        let output = linear.forward(&input)?;

        println!("Output without bias: {:?}", output.to_vec()?);
        Ok(())
    }

    #[test]
    fn test_linear_backward_with_bias() -> NnResult<()> {
        let device = Device::cpu();

        let mut input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        input.with_grad();

        let linear = Linear::new_with_device(3, 2, None, true, device)?;

        let output = linear.forward(&input)?;
        println!("Forward output: {:?}", output.to_vec()?);

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        println!("Input Gradients: {:?}", input_grad);

        assert_eq!(
            input_grad.len(),
            input.shape().iter().product(),
            "Input gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_linear_backward_without_bias() -> NnResult<()> {
        let device = Device::cpu();

        let mut input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        input.with_grad();

        let linear = Linear::new_with_device(3, 2, None, false, device)?;

        let output = linear.forward(&input)?;
        println!("Forward output: {:?}", output.to_vec()?);

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        println!("Input Gradients: {:?}", input_grad);

        assert_eq!(
            input_grad.len(),
            input.shape().iter().product(),
            "Input gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_linear_backward_with_custom_weight() -> NnResult<()> {
        let device = Device::cpu();

        let mut input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        input.with_grad();

        let weight =
            Tensor::from_vec_with_device(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3], &device)?;
        let bias = true;

        let linear = Linear::new_with_device(3, 2, Some(weight), bias, device)?;

        let output = linear.forward(&input)?;
        println!("Forward output: {:?}", output.to_vec()?);

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        println!("Input Gradients: {:?}", input_grad);

        assert_eq!(
            input_grad.len(),
            input.shape().iter().product(),
            "Input gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_linear_forward_with_bias() -> NnResult<()> {
        let device = Device::cuda(0);
        let input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        let linear = Linear::new_with_device(3, 2, None, true, device)?;
        let output = linear.forward(&input)?;

        println!("Output with bias: {:?}", output.to_vec()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_linear_forward_without_bias() -> NnResult<()> {
        let device = Device::cuda(0);
        let input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        let linear = Linear::new_with_device(3, 2, None, false, device)?;
        let output = linear.forward(&input)?;

        println!("Output without bias: {:?}", output.to_vec()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_linear_backward_with_bias() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        input.with_grad();

        let linear = Linear::new_with_device(3, 2, None, true, device)?;

        let output = linear.forward(&input)?;
        println!("Forward output: {:?}", output.to_vec()?);

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        println!("Input Gradients: {:?}", input_grad);

        assert_eq!(
            input_grad.len(),
            input.shape().iter().product(),
            "Input gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_linear_backward_without_bias() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        input.with_grad();

        let linear = Linear::new_with_device(3, 2, None, false, device)?;

        let output = linear.forward(&input)?;
        println!("Forward output: {:?}", output.to_vec()?);

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        println!("Input Gradients: {:?}", input_grad);

        assert_eq!(
            input_grad.len(),
            input.shape().iter().product(),
            "Input gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_linear_backward_with_custom_weight() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        input.with_grad();

        let weight =
            Tensor::from_vec_with_device(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3], &device)?;
        let bias = true;

        let linear = Linear::new_with_device(3, 2, Some(weight), bias, device)?;

        let output = linear.forward(&input)?;
        println!("Forward output: {:?}", output.to_vec()?);

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        println!("Input Gradients: {:?}", input_grad);

        assert_eq!(
            input_grad.len(),
            input.shape().iter().product(),
            "Input gradient shape mismatch"
        );

        Ok(())
    }
}
