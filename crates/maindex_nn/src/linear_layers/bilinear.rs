use crate::{error::NnResult, module::Module};
use maidenx_cpu::nn_layers::linear::{cpu_bilinear_backward, cpu_bilinear_forward};
#[cfg(feature = "cuda")]
use maidenx_cuda::nn_layers::linear::{cuda_bilinear_backward, cuda_bilinear_forward};
use maidenx_device::{get_current_device, Device};
use maidenx_tensor::{gradient::Node, tape::TENSOR_TAPE, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Module)]
#[module(inputs = 2)]
pub struct Bilinear {
    dim1: usize,
    dim2: usize,
    out_features: usize,
    weight: Tensor,
    device: Device,
}

impl Bilinear {
    pub fn new(
        dim1: usize,
        dim2: usize,
        out_features: usize,
        weight: Option<Tensor>,
    ) -> NnResult<Self> {
        let device = get_current_device();
        Self::new_with_device(dim1, dim2, out_features, weight, device)
    }

    pub fn new_with_device(
        dim1: usize,
        dim2: usize,
        out_features: usize,
        weight: Option<Tensor>,
        device: Device,
    ) -> NnResult<Self> {
        let mut weight = weight.unwrap_or_else(|| {
            let k = 1.0 / ((dim1 * dim2) as f32).sqrt();
            Tensor::randn_with_device(&[out_features, dim1, dim2], &device)
                .unwrap()
                .scalar_mul(k)
                .unwrap()
        });
        weight.with_grad();

        Ok(Self {
            dim1,
            dim2,
            out_features,
            weight,
            device,
        })
    }

    pub fn forward(&self, (input1, input2): (&Tensor, &Tensor)) -> NnResult<Tensor> {
        let batch_size = input1.shape()[0];
        let output_shape = &[batch_size, self.out_features];
        let mut output = Tensor::from_vec_with_device(
            vec![0.0; output_shape.iter().product()],
            output_shape,
            &self.device,
        )?;

        unsafe {
            match self.device {
                Device::Cpu => {
                    cpu_bilinear_forward(
                        output.buffer_mut().as_mut_ptr(),
                        input1.buffer().as_ptr(),
                        input2.buffer().as_ptr(),
                        self.weight.buffer().as_ptr(),
                        batch_size,
                        self.dim1,
                        self.dim2,
                        self.out_features,
                    )?;
                }
                #[cfg(feature = "cuda")]
                Device::Cuda(_) => {
                    cuda_bilinear_forward(
                        output.buffer_mut().as_mut_ptr(),
                        input1.buffer().as_ptr(),
                        input2.buffer().as_ptr(),
                        self.weight.buffer().as_ptr(),
                        batch_size,
                        self.dim1,
                        self.dim2,
                        self.out_features,
                        None,
                    )?;
                }
            }
        }

        let node = if input1.is_requires_grad() || input2.is_requires_grad() {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let input1 = input1.clone();
                    let input2 = input2.clone();
                    let weight = self.weight.clone();

                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut d_input1 =
                            Tensor::zeros_with_device(input1.shape(), input1.device()).unwrap();
                        let mut d_input2 =
                            Tensor::zeros_with_device(input2.shape(), input2.device()).unwrap();
                        let mut d_weights =
                            Tensor::zeros_with_device(weight.shape(), weight.device()).unwrap();

                        unsafe {
                            match input1.device() {
                                Device::Cpu => {
                                    cpu_bilinear_backward(
                                        grad_output.buffer().as_ptr(),
                                        input1.buffer().as_ptr(),
                                        input2.buffer().as_ptr(),
                                        weight.buffer().as_ptr(),
                                        d_input1.buffer_mut().as_mut_ptr(),
                                        d_input2.buffer_mut().as_mut_ptr(),
                                        d_weights.buffer_mut().as_mut_ptr(),
                                        input1.shape()[0],
                                        weight.shape()[1],
                                        weight.shape()[2],
                                        weight.shape()[0],
                                    )
                                    .unwrap();
                                }
                                #[cfg(feature = "cuda")]
                                Device::Cuda(_) => {
                                    cuda_bilinear_backward(
                                        grad_output.buffer().as_ptr(),
                                        input1.buffer().as_ptr(),
                                        input2.buffer().as_ptr(),
                                        weight.buffer().as_ptr(),
                                        d_input1.buffer_mut().as_mut_ptr(),
                                        d_input2.buffer_mut().as_mut_ptr(),
                                        d_weights.buffer_mut().as_mut_ptr(),
                                        input1.shape()[0],
                                        weight.shape()[1],
                                        weight.shape()[2],
                                        weight.shape()[0],
                                        None,
                                    )
                                    .unwrap();
                                }
                            }
                        }

                        vec![d_input1, d_input2, d_weights]
                    }
                })),
                inputs: vec![
                    input1
                        .node()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    input2
                        .node()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
                grad: None,
            })))
        } else {
            None
        };

        output.set_requires_grad(input1.is_requires_grad() || input2.is_requires_grad());
        output.set_node(node);

        TENSOR_TAPE.with(|tape| tape.borrow_mut().add(output.clone()));

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_device::Device;

    #[test]
    fn test_bilinear_forward() -> NnResult<()> {
        let device = Device::cpu();

        let input1 = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let input2 = Tensor::from_vec_with_device(vec![0.5, 0.6, 0.7, 0.8], &[2, 2], &device)?;

        let bilinear = Bilinear::new_with_device(2, 2, 1, None, device)?;

        let output = bilinear.forward((&input1, &input2))?;
        println!("Bilinear Forward Output: {:?}", output.to_vec()?);

        assert_eq!(output.shape(), &[2, 1], "Output shape mismatch");

        Ok(())
    }

    #[test]
    fn test_bilinear_backward() -> NnResult<()> {
        let device = Device::cpu();

        let mut input1 = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let mut input2 = Tensor::from_vec_with_device(vec![0.5, 0.6, 0.7, 0.8], &[2, 2], &device)?;
        input1.with_grad();
        input2.with_grad();

        let bilinear = Bilinear::new_with_device(2, 2, 1, None, device)?;

        let output = bilinear.forward((&input1, &input2))?;
        println!("Bilinear Forward Output: {:?}", output.to_vec()?);

        output.backward()?;

        let input1_grad = input1.grad()?.unwrap().to_vec()?;
        let input2_grad = input2.grad()?.unwrap().to_vec()?;
        println!("Input1 Gradients: {:?}", input1_grad);
        println!("Input2 Gradients: {:?}", input2_grad);

        assert_eq!(
            input1_grad.len(),
            input1.shape().iter().product(),
            "Input1 gradient shape mismatch"
        );
        assert_eq!(
            input2_grad.len(),
            input2.shape().iter().product(),
            "Input2 gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_bilinear_with_custom_weight() -> NnResult<()> {
        let device = Device::cpu();

        let mut input1 = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let mut input2 = Tensor::from_vec_with_device(vec![0.5, 0.6, 0.7, 0.8], &[2, 2], &device)?;
        input1.with_grad();
        input2.with_grad();

        let weight = Tensor::from_vec_with_device(vec![0.1, 0.2, 0.3, 0.4], &[1, 2, 2], &device)?;
        let bilinear = Bilinear::new_with_device(2, 2, 1, Some(weight), device)?;

        let output = bilinear.forward((&input1, &input2))?;
        println!("Bilinear Forward Output: {:?}", output.to_vec()?);

        output.backward()?;

        let input1_grad = input1.grad()?.unwrap().to_vec()?;
        let input2_grad = input2.grad()?.unwrap().to_vec()?;
        println!("Input1 Gradients: {:?}", input1_grad);
        println!("Input2 Gradients: {:?}", input2_grad);

        assert_eq!(
            input1_grad.len(),
            input1.shape().iter().product(),
            "Input1 gradient shape mismatch"
        );
        assert_eq!(
            input2_grad.len(),
            input2.shape().iter().product(),
            "Input2 gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_bilinear_forward() -> NnResult<()> {
        let device = Device::cuda(0);

        let input1 = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let input2 = Tensor::from_vec_with_device(vec![0.5, 0.6, 0.7, 0.8], &[2, 2], &device)?;

        let bilinear = Bilinear::new_with_device(2, 2, 1, None, device)?;

        let output = bilinear.forward((&input1, &input2))?;
        println!("Bilinear Forward Output: {:?}", output.to_vec()?);

        assert_eq!(output.shape(), &[2, 1], "Output shape mismatch");

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_bilinear_backward() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input1 = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let mut input2 = Tensor::from_vec_with_device(vec![0.5, 0.6, 0.7, 0.8], &[2, 2], &device)?;
        input1.with_grad();
        input2.with_grad();

        let bilinear = Bilinear::new_with_device(2, 2, 1, None, device)?;

        let output = bilinear.forward((&input1, &input2))?;
        println!("Bilinear Forward Output: {:?}", output.to_vec()?);

        output.backward()?;

        let input1_grad = input1.grad()?.unwrap().to_vec()?;
        let input2_grad = input2.grad()?.unwrap().to_vec()?;
        println!("Input1 Gradients: {:?}", input1_grad);
        println!("Input2 Gradients: {:?}", input2_grad);

        assert_eq!(
            input1_grad.len(),
            input1.shape().iter().product(),
            "Input1 gradient shape mismatch"
        );
        assert_eq!(
            input2_grad.len(),
            input2.shape().iter().product(),
            "Input2 gradient shape mismatch"
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_bilinear_with_custom_weight() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut input1 = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let mut input2 = Tensor::from_vec_with_device(vec![0.5, 0.6, 0.7, 0.8], &[2, 2], &device)?;
        input1.with_grad();
        input2.with_grad();

        let weight = Tensor::from_vec_with_device(vec![0.1, 0.2, 0.3, 0.4], &[1, 2, 2], &device)?;
        let bilinear = Bilinear::new_with_device(2, 2, 1, Some(weight), device)?;

        let output = bilinear.forward((&input1, &input2))?;
        println!("Bilinear Forward Output: {:?}", output.to_vec()?);

        output.backward()?;

        let input1_grad = input1.grad()?.unwrap().to_vec()?;
        let input2_grad = input2.grad()?.unwrap().to_vec()?;
        println!("Input1 Gradients: {:?}", input1_grad);
        println!("Input2 Gradients: {:?}", input2_grad);

        assert_eq!(
            input1_grad.len(),
            input1.shape().iter().product(),
            "Input1 gradient shape mismatch"
        );
        assert_eq!(
            input2_grad.len(),
            input2.shape().iter().product(),
            "Input2 gradient shape mismatch"
        );

        Ok(())
    }
}
