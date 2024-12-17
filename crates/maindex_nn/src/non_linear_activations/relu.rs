use crate::{error::NnResult, module::Module};
use maidenx_cpu::nn_activations::non_linear::{cpu_relu_backward, cpu_relu_forward};
#[cfg(feature = "cuda")]
use maidenx_cuda::nn_activations::non_linear::{cuda_relu_backward, cuda_relu_forward};
use maidenx_device::Device;
use maidenx_tensor::{gradient::Node, tape::TENSOR_TAPE, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Default, Module)]
pub struct ReLU {}

impl ReLU {
    pub fn new() -> Self {
        ReLU::default()
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let device = input.device();
        let mut output =
            Tensor::from_vec_with_device(vec![0.0; input.size()], input.shape(), device)?;

        match device {
            Device::Cpu => unsafe {
                cpu_relu_forward(
                    output.buffer_mut().as_mut_ptr(),
                    input.buffer().as_ptr(),
                    input.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_relu_forward(
                    output.buffer_mut().as_mut_ptr(),
                    input.buffer().as_ptr(),
                    input.size(),
                    None,
                )?;
            },
        }

        let node = if input.is_requires_grad() {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let input = input.clone();

                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let device = input.device();
                        let mut grad_input = Tensor::from_vec_with_device(
                            vec![0.0; input.size()],
                            input.shape(),
                            device,
                        )
                        .unwrap();

                        match device {
                            Device::Cpu => unsafe {
                                cpu_relu_backward(
                                    grad_input.buffer_mut().as_mut_ptr(),
                                    grad_output.buffer().as_ptr(),
                                    input.buffer().as_ptr(),
                                    input.size(),
                                )
                                .unwrap();
                            },
                            #[cfg(feature = "cuda")]
                            Device::Cuda(_) => unsafe {
                                cuda_relu_backward(
                                    grad_input.buffer_mut().as_mut_ptr(),
                                    grad_output.buffer().as_ptr(),
                                    input.buffer().as_ptr(),
                                    input.size(),
                                    None,
                                )
                                .unwrap();
                            },
                        }

                        vec![grad_input]
                    }
                })),
                inputs: vec![input.node().map(|n| Arc::downgrade(&n)).unwrap_or_default()],
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
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_relu() -> NnResult<()> {
        let device = Device::cpu();

        let input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        let relu = ReLU::default();
        let output = relu.forward(&input)?;
        assert_eq!(output.to_vec()?, vec![0.0, 0.0, 1.0, 2.0]);

        let input = Tensor::from_device(vec![vec![-2.0, 0.0], vec![1.0, -3.0]], &device)?;
        let output = relu.forward(&input)?;
        assert_eq!(output.to_vec()?, vec![0.0, 0.0, 1.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_relu_backward() -> NnResult<()> {
        let device = Device::Cpu;

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let relu = ReLU::default();
        let output = relu.forward(&input)?;
        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        assert_eq!(input_grad, vec![0.0, 0.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_relu_chain_backward() -> NnResult<()> {
        let device = Device::Cpu;

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let relu1 = ReLU::default();
        let relu2 = ReLU::default();

        let hidden = relu1.forward(&input)?;
        let output = relu2.forward(&hidden)?;

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        assert_eq!(input_grad, vec![0.0, 0.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_relu() -> NnResult<()> {
        let device = Device::cuda(0);

        let input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        let relu = ReLU::default();
        let output = relu.forward(&input)?;
        assert_eq!(output.to_vec()?, vec![0.0, 0.0, 1.0, 2.0]);

        let input = Tensor::from_device(vec![vec![-2.0, 0.0], vec![1.0, -3.0]], &device)?;
        let output = relu.forward(&input)?;
        assert_eq!(output.to_vec()?, vec![0.0, 0.0, 1.0, 0.0]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_relu_backward() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let relu = ReLU::default();
        let output = relu.forward(&input)?;
        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        assert_eq!(input_grad, vec![0.0, 0.0, 1.0, 1.0]);

        Ok(())
    }
}
