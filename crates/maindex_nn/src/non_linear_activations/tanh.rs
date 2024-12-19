use crate::{error::NnResult, module::Module};
use maidenx_cpu::nn_activations::non_linear::{cpu_tanh_backward, cpu_tanh_forward};
#[cfg(feature = "cuda")]
use maidenx_cuda::nn_activations::non_linear::{cuda_tanh_backward, cuda_tanh_forward};
use maidenx_device::Device;
use maidenx_tensor::{gradient::Node, tape::TENSOR_TAPE, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Default, Module)]
pub struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        Tanh::default()
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let device = input.device();
        let mut output =
            Tensor::from_vec_with_device(vec![0.0; input.size()], input.shape(), device)?;

        match device {
            Device::Cpu => unsafe {
                cpu_tanh_forward(
                    output.buffer_mut().as_mut_ptr(),
                    input.buffer().as_ptr(),
                    input.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tanh_forward(
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

                        let mut output = Tensor::from_vec_with_device(
                            vec![0.0; input.size()],
                            input.shape(),
                            device,
                        )
                        .unwrap();

                        match device {
                            Device::Cpu => unsafe {
                                cpu_tanh_forward(
                                    output.buffer_mut().as_mut_ptr(),
                                    input.buffer().as_ptr(),
                                    input.size(),
                                )
                                .unwrap();

                                cpu_tanh_backward(
                                    grad_input.buffer_mut().as_mut_ptr(),
                                    grad_output.buffer().as_ptr(),
                                    output.buffer().as_ptr(),
                                    input.size(),
                                )
                                .unwrap();
                            },
                            #[cfg(feature = "cuda")]
                            Device::Cuda(_) => unsafe {
                                cuda_tanh_forward(
                                    output.buffer_mut().as_mut_ptr(),
                                    input.buffer().as_ptr(),
                                    input.size(),
                                    None,
                                )
                                .unwrap();

                                cuda_tanh_backward(
                                    grad_input.buffer_mut().as_mut_ptr(),
                                    grad_output.buffer().as_ptr(),
                                    output.buffer().as_ptr(),
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
    fn cpu_tanh_forward() -> NnResult<()> {
        let device = Device::cpu();

        let input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        let tanh = Tanh::default();
        let output = tanh.forward(&input)?;
        // tanh values
        assert!((output.to_vec()?[0] - (-0.7616)).abs() < 1e-4); // tanh(-1)
        assert!((output.to_vec()?[1] - 0.0000).abs() < 1e-4); // tanh(0)
        assert!((output.to_vec()?[2] - 0.7616).abs() < 1e-4); // tanh(1)
        assert!((output.to_vec()?[3] - 0.9640).abs() < 1e-4); // tanh(2)

        let input = Tensor::from_device(vec![vec![-2.0, 0.0], vec![1.0, -3.0]], &device)?;
        let output = tanh.forward(&input)?;
        let result = output.to_vec()?;
        assert!((result[0] - (-0.9640)).abs() < 1e-4); // tanh(-2)
        assert!((result[1] - 0.0000).abs() < 1e-4); // tanh(0)
        assert!((result[2] - 0.7616).abs() < 1e-4); // tanh(1)
        assert!((result[3] - (-0.9951)).abs() < 1e-4); // tanh(-3)

        Ok(())
    }

    #[test]
    fn cpu_tanh_backward() -> NnResult<()> {
        let device = Device::Cpu;

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let tanh = Tanh::default();
        let output = tanh.forward(&input)?;
        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        // tanh gradient: 1 - tanh²(x)
        assert!((input_grad[0] - 0.4200).abs() < 1e-4); // grad at x=-1
        assert!((input_grad[1] - 1.0000).abs() < 1e-4); // grad at x=0
        assert!((input_grad[2] - 0.4200).abs() < 1e-4); // grad at x=1
        assert!((input_grad[3] - 0.0707).abs() < 1e-4); // grad at x=2

        Ok(())
    }

    #[test]
    fn cpu_tanh_chain_backward() -> NnResult<()> {
        let device = Device::Cpu;

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let tanh1 = Tanh::default();
        let tanh2 = Tanh::default();

        let hidden = tanh1.forward(&input)?;
        let output = tanh2.forward(&hidden)?;

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        // Checking if the chain rule is working
        assert!(input_grad.iter().all(|&x| x != 0.0));

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_tanh_forward() -> NnResult<()> {
        let device = Device::cuda(0);

        let input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        let tanh = Tanh::default();
        let output = tanh.forward(&input)?;
        let result = output.to_vec()?;
        assert!((result[0] - (-0.7616)).abs() < 1e-4);
        assert!((result[1] - 0.0000).abs() < 1e-4);
        assert!((result[2] - 0.7616).abs() < 1e-4);
        assert!((result[3] - 0.9640).abs() < 1e-4);

        let input = Tensor::from_device(vec![vec![-2.0, 0.0], vec![1.0, -3.0]], &device)?;
        let output = tanh.forward(&input)?;
        let result = output.to_vec()?;
        assert!((result[0] - (-0.9640)).abs() < 1e-4);
        assert!((result[1] - 0.0000).abs() < 1e-4);
        assert!((result[2] - 0.7616).abs() < 1e-4);
        assert!((result[3] - (-0.9951)).abs() < 1e-4);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_tanh_backward() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let tanh = Tanh::default();
        let output = tanh.forward(&input)?;
        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        assert!((input_grad[0] - 0.4200).abs() < 1e-4);
        assert!((input_grad[1] - 1.0000).abs() < 1e-4);
        assert!((input_grad[2] - 0.4200).abs() < 1e-4);
        assert!((input_grad[3] - 0.0707).abs() < 1e-4);

        Ok(())
    }
}
