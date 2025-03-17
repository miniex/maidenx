#![allow(clippy::too_many_arguments)]

use crate::layer::{Layer, LayerState};
use maidenx_core::{
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::{Error, Result},
};
use maidenx_tensor::{Tensor, TensorNode};

// #[derive(Layer, Clone)]
// pub struct Conv1d {
//     weight: Tensor,
//     bias: Option<Tensor>,
//     kernel_size: usize,
//     stride: usize,
//     padding: usize,
// }
//
// impl Conv1d {
//     pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, with_bias: bool) -> Result<Self> {
//         let device = get_default_device();
//         let dtype = get_default_dtype();
//
//         Self::new_with_spec(in_channels, out_channels, kernel_size, stride, padding, with_bias, device, dtype)
//     }
//
//     pub fn new_with_spec(
//         in_channels: usize,
//         out_channels: usize,
//         kernel_size: usize,
//         stride: usize,
//         padding: usize,
//         with_bias: bool,
//         device: Device,
//         dtype: DType,
//     ) -> Result<Self> {
//         let k: f32 = 1.0 / ((in_channels * kernel_size) as f32).sqrt();
//
//         let mut w = Tensor::randn_with_spec(&[out_channels, in_channels, kernel_size], device, dtype)?;
//         w.with_grad()?;
//         w.mul_scalar(k)?;
//
//         let b = if with_bias {
//             let mut b = Tensor::randn_with_spec(&[out_channels], device, dtype)?;
//             b.with_grad()?;
//             b.mul_scalar(k)?;
//             Some(b)
//         } else {
//             None
//         };
//
//         Ok(Self {
//             weight: w,
//             bias: b,
//             kernel_size,
//             stride,
//             padding,
//         })
//     }
//
//     pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
//         let input_shape = input.shape();
//         if input_shape.len() != 3 {
//             return Err(maidenx_core::error::Error::InvalidShape {
//                 message: "Expected 3D input (batch_size, channels, length)".to_string(),
//             });
//         }
//
//         let batch_size = input_shape[0];
//         let in_channels = input_shape[1];
//         let input_length = input_shape[2];
//
//         let output_length = (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
//
//         let weight_reshaped = self.weight.reshape(&[self.weight.shape()[0], -1])?;
//         let mut output = input
//             .unfold(2, self.kernel_size, self.stride)?
//             .matmul(&weight_reshaped.transpose(-1, -2)?)?;
//
//         if let Some(ref bias) = self.bias {
//             output = output.add(bias)?;
//         }
//
//         Ok(output)
//     }
//
//     pub fn parameters(&mut self) -> Vec<&mut Tensor> {
//         let mut params = vec![&mut self.weight];
//         if let Some(ref mut b) = self.bias {
//             params.push(b);
//         }
//         params
//     }
//
//     pub fn weight(&self) -> &Tensor {
//         &self.weight
//     }
//
//     pub fn bias(&self) -> Option<&Tensor> {
//         self.bias.as_ref()
//     }
// }

#[derive(Layer, Clone)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),

    state: LayerState,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        with_bias: bool,
    ) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::new_with_spec(in_channels, out_channels, kernel_size, stride, padding, with_bias, device, dtype)
    }

    pub fn new_with_spec(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        with_bias: bool,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let k: f32 = 1.0 / ((in_channels * kernel_size.0 * kernel_size.1) as f32).sqrt();

        // weight
        let mut w = Tensor::randn_with_spec(&[out_channels, in_channels, kernel_size.0, kernel_size.1], device, dtype)?;
        w.with_grad()?;
        w.mul_scalar(k)?;

        // bias
        let b = if with_bias {
            let mut b = Tensor::randn_with_spec(&[out_channels], device, dtype)?;
            b.with_grad()?;
            b.mul_scalar(k)?;

            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight: w,
            bias: b,
            kernel_size,
            stride,
            padding,
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(Error::InvalidShape {
                message: "Expected 4D input (batch_size, channels, height, width)".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let output_height = (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let cols = conv2d::im2col(
            input,
            batch_size,
            in_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            self.kernel_size,
            self.stride,
            self.padding,
        )?;

        let weight_matrix = self.weight.reshape(&[
            self.weight.shape()[0],
            self.weight.shape()[1] * self.weight.shape()[2] * self.weight.shape()[3],
        ])?;

        let mut output = cols.matmul(&weight_matrix.transpose(-1, -2)?)?;

        output = output.reshape(&[batch_size as isize, output_height as isize, output_width as isize, -1])?;
        output = output.transpose(1, 3)?.transpose(2, 3)?;

        if let Some(ref bias) = self.bias {
            let bias_shape = vec![1, bias.shape()[0], 1, 1];
            let bias = bias.reshape(&bias_shape)?;
            output = output.add(&bias)?;
        }

        if input.requires_grad() {
            let kernel_size = self.kernel_size;
            let stride = self.stride;
            let padding = self.padding;
            let cols = cols;

            let backward_fn = Box::new(move |inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let input = &inputs[0];
                let weight = &inputs[1];
                let mut grads = Vec::new();

                if input.requires_grad() {
                    let weight_matrix = weight.reshape(&[weight.shape()[0], weight.shape()[1] * weight.shape()[2] * weight.shape()[3]])?;

                    let grad_reshaped = grad_out.transpose(2, 3)?.transpose(1, 3)?;
                    let grad_flat = grad_reshaped.reshape(&[grad_out.shape()[0] * output_height * output_width, grad_out.shape()[1]])?;

                    let grad_cols = grad_flat.matmul(&weight_matrix)?;

                    let grad_input = conv2d::col2im(
                        &grad_cols,
                        batch_size,
                        in_channels,
                        input_height,
                        input_width,
                        output_height,
                        output_width,
                        kernel_size,
                        stride,
                        padding,
                    )?;

                    grads.push(grad_input);
                } else {
                    grads.push(Tensor::zeros_like(input)?);
                }

                if weight.requires_grad() {
                    let grad_reshaped = grad_out.transpose(2, 3)?.transpose(1, 3)?;
                    let grad_flat = grad_reshaped.reshape(&[grad_out.shape()[0] * output_height * output_width, grad_out.shape()[1]])?;

                    let grad_weight = cols.transpose(-1, -2)?.matmul(&grad_flat)?;
                    let grad_weight = grad_weight.reshape(&[weight.shape()[0], weight.shape()[1], kernel_size.0, kernel_size.1])?;
                    grads.push(grad_weight);
                } else {
                    grads.push(Tensor::zeros_like(weight)?);
                }

                if inputs.len() > 2 {
                    let bias = &inputs[2];
                    if bias.requires_grad() {
                        let mut grad_bias = grad_out.clone();
                        grad_bias = grad_bias.sum(3, false)?;
                        grad_bias = grad_bias.sum(2, false)?;
                        grad_bias = grad_bias.sum(0, false)?;
                        grads.push(grad_bias);
                    } else {
                        grads.push(Tensor::zeros_like(bias)?);
                    }
                }

                Ok(grads)
            });

            let mut inputs = vec![input.clone(), self.weight.clone()];
            if let Some(ref bias) = self.bias {
                inputs.push(bias.clone());
            }

            let node = TensorNode::new("conv2d".to_string(), inputs, Some(backward_fn));
            output.set_node(node);
        }

        Ok(output)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![];
        params.push(&mut self.weight);
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

mod conv2d {
    use super::*;

    pub fn im2col(
        input: &Tensor,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        output_height: usize,
        output_width: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let cols_shape = [batch_size * output_height * output_width, channels * kernel_size.0 * kernel_size.1];
        let mut result = Tensor::empty_with_spec(&cols_shape, input.device(), input.dtype())?;

        let num_els = cols_shape.iter().product();
        let dims_and_strides = vec![
            batch_size,
            channels,
            height,
            width,
            kernel_size.0,
            kernel_size.1,
            output_height,
            output_width,
            stride.0,
            stride.1,
            padding.0,
            padding.1,
        ];

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::nn::conv::im2col(out_buf, input.buffer(), num_els, Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn col2im(
        input: &Tensor,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        output_height: usize,
        output_width: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let output_shape = [batch_size, channels, height, width];
        let mut result = Tensor::empty_with_spec(&output_shape, input.device(), input.dtype())?;

        let num_els = output_shape.iter().product();
        let dims_and_strides = vec![
            batch_size,
            channels,
            height,
            width,
            kernel_size.0,
            kernel_size.1,
            output_height,
            output_width,
            stride.0,
            stride.1,
            padding.0,
            padding.1,
        ];

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::nn::conv::col2im(out_buf, input.buffer(), num_els, Some(&dims_and_strides))?;

                Ok(())
            })?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_core::device::set_default_device;

    fn setup_device() {
        #[cfg(feature = "cuda")]
        set_default_device(Device::CUDA(0));
        #[cfg(not(any(feature = "cuda")))]
        set_default_device(Device::CPU);
    }

    #[test]
    fn conv2d_forward() -> Result<()> {
        setup_device();

        let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), true)?;

        let input = Tensor::randn(&[1, 3, 32, 32])?;
        let output = conv.forward(&input)?;
        assert_eq!(output.shape(), &[1, 64, 32, 32]);

        Ok(())
    }

    #[test]
    fn conv2d_backward() -> Result<()> {
        setup_device();

        let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), true)?;

        let mut input = Tensor::randn(&[1, 3, 32, 32])?;
        input.with_grad()?;
        let output = conv.forward(&input)?;
        let loss = output.sum_all()?;
        loss.backward()?;

        let input_grad = input.grad()?.expect("Input gradient should exist");
        assert_eq!(input_grad.shape(), &[1, 3, 32, 32]);

        let weight_grad = conv.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[64, 3, 3, 3]);

        if let Some(bias) = conv.bias() {
            let bias_grad = bias.grad()?.expect("Bias gradient should exist");
            assert_eq!(bias_grad.shape(), &[64]);
        }

        Ok(())
    }
}
