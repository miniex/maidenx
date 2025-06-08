use crate::Tensor;
use maidenx_core::dtype::DType;
use std::fmt;

const MAX_ELEMENTS_PER_DIM: usize = 4;

fn display_tensor_data<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    stride: usize,
    shape: &[usize],
    depth: usize,
    use_float_format: bool,
    column_width: Option<usize>,
) -> fmt::Result {
    match shape.len() {
        0 => {
            if use_float_format {
                write!(f, "{:.6}", data[0])
            } else {
                write!(f, "{}", data[0])
            }
        },
        1 => {
            write!(f, "[")?;
            let len = data.len();
            let (show_start, show_end) = if len <= MAX_ELEMENTS_PER_DIM * 2 {
                (len, 0)
            } else {
                (MAX_ELEMENTS_PER_DIM, MAX_ELEMENTS_PER_DIM)
            };

            let width = column_width.unwrap_or_else(|| {
                data.iter()
                    .map(|val| {
                        if use_float_format {
                            format!("{:.6}", val).len()
                        } else {
                            format!("{}", val).len()
                        }
                    })
                    .max()
                    .unwrap_or(0)
            });

            for i in 0..show_start.min(len) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let formatted = if use_float_format {
                    format!("{:.6}", &data[i])
                } else {
                    format!("{}", &data[i])
                };
                write!(f, "{:width$}", formatted, width = width)?;
            }

            if len > show_start + show_end {
                write!(f, ", ... ")?;
            }

            if show_end > 0 && len > show_start + show_end {
                for i in (len - show_end)..len {
                    write!(f, ", ")?;
                    let formatted = if use_float_format {
                        format!("{:.6}", &data[i])
                    } else {
                        format!("{}", &data[i])
                    };
                    write!(f, "{:width$}", formatted, width = width)?;
                }
            }
            write!(f, "]")
        },
        _ => {
            let sub_stride = stride / shape[0];
            write!(f, "[")?;
            let dim_size = shape[0];
            let (show_start, show_end) = if dim_size <= MAX_ELEMENTS_PER_DIM * 2 {
                (dim_size, 0)
            } else {
                (MAX_ELEMENTS_PER_DIM, MAX_ELEMENTS_PER_DIM)
            };

            let width = if shape.len() == 2 {
                let mut max_width = 0;
                for i in 0..shape[0].min(3) {
                    for j in 0..shape[1].min(3) {
                        let idx = i * sub_stride + j;
                        if idx < data.len() {
                            let formatted = if use_float_format {
                                format!("{:.6}", &data[idx])
                            } else {
                                format!("{}", &data[idx])
                            };
                            max_width = max_width.max(formatted.len());
                        }
                    }
                }
                Some(max_width)
            } else {
                None
            };

            for i in 0..show_start.min(dim_size) {
                if i > 0 {
                    if depth < 2 {
                        write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    } else {
                        write!(f, ", ")?;
                    }
                }
                display_tensor_data(
                    f,
                    &data[i * sub_stride..(i + 1) * sub_stride],
                    sub_stride,
                    &shape[1..],
                    depth + 1,
                    use_float_format,
                    width,
                )?;
            }

            if dim_size > show_start + show_end {
                if depth < 2 {
                    write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    write!(f, "...")?;
                } else {
                    write!(f, ", ... ")?;
                }
            }

            if show_end > 0 && dim_size > show_start + show_end {
                for i in (dim_size - show_end)..dim_size {
                    if depth < 2 {
                        write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    } else {
                        write!(f, ", ")?;
                    }
                    display_tensor_data(
                        f,
                        &data[i * sub_stride..(i + 1) * sub_stride],
                        sub_stride,
                        &shape[1..],
                        depth + 1,
                        use_float_format,
                        width,
                    )?;
                }
            }
            write!(f, "]")
        },
    }
}

fn debug_format_data<T: fmt::Display>(data: &[T], use_float_format: bool) -> String {
    let mut result = String::from("[");
    let len = data.len();

    let (show_start, show_end) = if len <= MAX_ELEMENTS_PER_DIM * 2 {
        (len, 0)
    } else {
        (MAX_ELEMENTS_PER_DIM, MAX_ELEMENTS_PER_DIM)
    };

    for i in 0..show_start.min(len) {
        if i > 0 {
            result.push_str(", ");
        }
        if use_float_format {
            result.push_str(&format!("{:.6}", &data[i]));
        } else {
            result.push_str(&format!("{}", &data[i]));
        }
    }

    if len > show_start + show_end {
        result.push_str(", ... ");
    }

    if show_end > 0 && len > show_start + show_end {
        for i in (len - show_end)..len {
            result.push_str(", ");
            if use_float_format {
                result.push_str(&format!("{:.6}", &data[i]));
            } else {
                result.push_str(&format!("{}", &data[i]));
            }
        }
    }

    result.push(']');
    result
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dtype() {
            DType::BF16 | DType::F16 => {
                let data = self.to_flatten_vec::<f32>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, true, None)
            },
            DType::F32 => {
                let data = self.to_flatten_vec::<f32>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, true, None)
            },
            DType::F64 => {
                let data = self.to_flatten_vec::<f64>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, true, None)
            },
            DType::BOOL => {
                let data = self.to_flatten_vec::<bool>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::U8 => {
                let data = self.to_flatten_vec::<u8>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::U16 => {
                let data = self.to_flatten_vec::<u16>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::U32 => {
                let data = self.to_flatten_vec::<u32>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::U64 => {
                let data = self.to_flatten_vec::<u64>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::I8 => {
                let data = self.to_flatten_vec::<i8>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::I16 => {
                let data = self.to_flatten_vec::<i16>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::I32 => {
                let data = self.to_flatten_vec::<i32>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
            DType::I64 => {
                let data = self.to_flatten_vec::<i64>();
                display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
            },
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape=[")?;
        let shape = self.shape();
        for (i, dim) in shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "], device={}, dtype={}", self.device().name(), self.dtype().as_str())?;
        write!(f, ", data=")?;

        match self.dtype() {
            DType::BF16 | DType::F16 => {
                let data = self.to_flatten_vec::<f32>();
                write!(f, "{}", debug_format_data(&data, true))?;
            },
            DType::F32 => {
                let data = self.to_flatten_vec::<f32>();
                write!(f, "{}", debug_format_data(&data, true))?;
            },
            DType::F64 => {
                let data = self.to_flatten_vec::<f64>();
                write!(f, "{}", debug_format_data(&data, true))?;
            },
            DType::BOOL => {
                let data = self.to_flatten_vec::<bool>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::U8 => {
                let data = self.to_flatten_vec::<u8>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::U16 => {
                let data = self.to_flatten_vec::<u16>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::U32 => {
                let data = self.to_flatten_vec::<u32>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::U64 => {
                let data = self.to_flatten_vec::<u64>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::I8 => {
                let data = self.to_flatten_vec::<i8>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::I16 => {
                let data = self.to_flatten_vec::<i16>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::I32 => {
                let data = self.to_flatten_vec::<i32>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
            DType::I64 => {
                let data = self.to_flatten_vec::<i64>();
                write!(f, "{}", debug_format_data(&data, false))?;
            },
        }

        write!(f, ", requires_grad={}", self.requires_grad())?;
        write!(f, ")")
    }
}
