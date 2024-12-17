use crate::{error::NnResult, module::Module};
use maidenx_cpu::nn_activations::non_linear::{cpu_sigmoid_backward, cpu_sigmoid_forward};
#[cfg(feature = "cuda")]
use maidenx_cuda::nn_activations::non_linear::{cuda_sigmoid_backward, cuda_sigmoid_forward};
use maidenx_device::Device;
use maidenx_tensor::{gradient::Node, tape::TENSOR_TAPE, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Default, Module)]
pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid::default()
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let device = input.device();
        let mut output =
            Tensor::from_vec_with_device(vec![0.0; input.size()], input.shape(), device)?;

        match device {
            Device::Cpu => unsafe {
                cpu_sigmoid_forward(
                    output.buffer_mut().as_mut_ptr(),
                    input.buffer().as_ptr(),
                    input.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_sigmoid_forward(
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
                                cpu_sigmoid_forward(
                                    output.buffer_mut().as_mut_ptr(),
                                    input.buffer().as_ptr(),
                                    input.size(),
                                )
                                .unwrap();

                                cpu_sigmoid_backward(
                                    grad_input.buffer_mut().as_mut_ptr(),
                                    grad_output.buffer().as_ptr(),
                                    output.buffer().as_ptr(),
                                    input.size(),
                                )
                                .unwrap();
                            },
                            #[cfg(feature = "cuda")]
                            Device::Cuda(_) => unsafe {
                                cuda_sigmoid_forward(
                                    output.buffer_mut().as_mut_ptr(),
                                    input.buffer().as_ptr(),
                                    input.size(),
                                    None,
                                )
                                .unwrap();

                                cuda_sigmoid_backward(
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
    fn test_cpu_sigmoid() -> NnResult<()> {
        let device = Device::cpu();

        let input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        let sigmoid = Sigmoid::default();
        let output = sigmoid.forward(&input)?;
        // Sigmoid values: 1/(1 + e^-x)
        assert!((output.to_vec()?[0] - 0.2689).abs() < 1e-4); // sigmoid(-1)
        assert!((output.to_vec()?[1] - 0.5000).abs() < 1e-4); // sigmoid(0)
        assert!((output.to_vec()?[2] - 0.7311).abs() < 1e-4); // sigmoid(1)
        assert!((output.to_vec()?[3] - 0.8808).abs() < 1e-4); // sigmoid(2)

        let input = Tensor::from_device(vec![vec![-2.0, 0.0], vec![1.0, -3.0]], &device)?;
        let output = sigmoid.forward(&input)?;
        let result = output.to_vec()?;
        assert!((result[0] - 0.1192).abs() < 1e-4); // sigmoid(-2)
        assert!((result[1] - 0.5000).abs() < 1e-4); // sigmoid(0)
        assert!((result[2] - 0.7311).abs() < 1e-4); // sigmoid(1)
        assert!((result[3] - 0.0474).abs() < 1e-4); // sigmoid(-3)

        Ok(())
    }

    #[test]
    fn test_sigmoid_backward() -> NnResult<()> {
        let device = Device::Cpu;

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let sigmoid = Sigmoid::default();
        let output = sigmoid.forward(&input)?;
        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
        assert!((input_grad[0] - 0.1966).abs() < 1e-4); // grad at x=-1
        assert!((input_grad[1] - 0.2500).abs() < 1e-4); // grad at x=0
        assert!((input_grad[2] - 0.1966).abs() < 1e-4); // grad at x=1
        assert!((input_grad[3] - 0.1050).abs() < 1e-4); // grad at x=2

        Ok(())
    }

    #[test]
    fn test_sigmoid_chain_backward() -> NnResult<()> {
        let device = Device::Cpu;

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let sigmoid1 = Sigmoid::default();
        let sigmoid2 = Sigmoid::default();

        let hidden = sigmoid1.forward(&input)?;
        let output = sigmoid2.forward(&hidden)?;

        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        // Checking if the chain rule is working
        assert!(input_grad.iter().all(|&x| x != 0.0));

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_sigmoid() -> NnResult<()> {
        let device = Device::cuda(0);

        let input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        let sigmoid = Sigmoid::default();
        let output = sigmoid.forward(&input)?;
        let result = output.to_vec()?;
        assert!((result[0] - 0.2689).abs() < 1e-4);
        assert!((result[1] - 0.5000).abs() < 1e-4);
        assert!((result[2] - 0.7311).abs() < 1e-4);
        assert!((result[3] - 0.8808).abs() < 1e-4);

        let input = Tensor::from_device(vec![vec![-2.0, 0.0], vec![1.0, -3.0]], &device)?;
        let output = sigmoid.forward(&input)?;
        let result = output.to_vec()?;
        assert!((result[0] - 0.1192).abs() < 1e-4);
        assert!((result[1] - 0.5000).abs() < 1e-4);
        assert!((result[2] - 0.7311).abs() < 1e-4);
        assert!((result[3] - 0.0474).abs() < 1e-4);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_sigmoid_backward() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input = Tensor::from_device(vec![-1.0, 0.0, 1.0, 2.0], &device)?;
        input.with_grad();

        let sigmoid = Sigmoid::default();
        let output = sigmoid.forward(&input)?;
        output.backward()?;

        let input_grad = input.grad()?.unwrap().to_vec()?;
        assert!((input_grad[0] - 0.1966).abs() < 1e-4);
        assert!((input_grad[1] - 0.2500).abs() < 1e-4);
        assert!((input_grad[2] - 0.1966).abs() < 1e-4);
        assert!((input_grad[3] - 0.1050).abs() < 1e-4);

        Ok(())
    }
}
