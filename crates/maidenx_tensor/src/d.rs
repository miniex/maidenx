use crate::Tensor;
use maidenx_core::dtype::DType;
use std::fmt;

macro_rules! impl_display_for_type {
    ($val_type:ty, $format:expr) => {
        fn display_tensor_data(f: &mut fmt::Formatter<'_>, data: &[$val_type], stride: usize, shape: &[usize], depth: usize) -> fmt::Result {
            match shape.len() {
                0 => write!(f, "{}", data[0]),
                1 => {
                    write!(f, "[")?;
                    for (i, val) in data.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?
                        }
                        write!(f, $format, val)?;
                    }
                    write!(f, "]")
                }
                _ => {
                    let sub_stride = stride / shape[0];
                    write!(f, "[")?;
                    for i in 0..shape[0] {
                        display_tensor_data(f, &data[i * sub_stride..(i + 1) * sub_stride], sub_stride, &shape[1..], depth + 1)?;
                        if i < shape[0] - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    write!(f, "]")
                }
            }
        }
    };
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        macro_rules! handle_type {
            ($type:ty, $format:expr) => {{
                if let Ok(data) = self.to_flatten_vec::<$type>() {
                    impl_display_for_type!($type, $format);
                    display_tensor_data(f, &data, self.size(), self.shape(), 0)
                } else {
                    write!(f, "Failed to fetch data")
                }
            }};
        }

        match self.dtype() {
            DType::BF16 | DType::F16 => handle_type!(f32, "{:.8}"),
            DType::F32 => handle_type!(f32, "{:.8}"),
            DType::F64 => handle_type!(f64, "{:.8}"),
            DType::BOOL => handle_type!(bool, "{}"),
            DType::U8 => handle_type!(u8, "{}"),
            DType::U32 => handle_type!(u32, "{}"),
            DType::I8 => handle_type!(i8, "{}"),
            DType::I32 => handle_type!(i32, "{}"),
            DType::I64 => handle_type!(i64, "{}"),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape=[")?;

        // Write shape
        let shape = self.shape();
        for (i, dim) in shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?
            }
            write!(f, "{}", dim)?;
        }

        write!(f, "], device={}, dtype={}", self.device().name(), self.dtype().as_str())?;

        write!(f, ", data=")?;
        fmt::Display::fmt(self, f)?;

        write!(f, ", requires_grad={}", self.requires_grad())?;

        if self.requires_grad() {
            match self.grad() {
                Ok(Some(grad)) => {
                    write!(f, ", grad=")?;
                    fmt::Display::fmt(&grad, f)?;
                }
                Ok(None) => write!(f, ", grad=None")?,
                Err(_) => write!(f, ", grad=<locked>")?,
            }
        }

        write!(f, ")")
    }
}
