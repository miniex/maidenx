use crate::Tensor;
use maidenx_core::dtype::DType;
use std::fmt;

const MAX_ELEMENTS_PER_DIM: usize = 4;

fn format_float<T: fmt::Display + Copy>(value: T) -> String
where
    T: Into<f64>,
{
    let val = value.into();
    if val.fract() == 0.0 {
        format!("{:.0}.", val)
    } else {
        let formatted = format!("{:.6}", val);
        let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
        if trimmed.contains('.') {
            trimmed.to_string()
        } else {
            format!("{}.", trimmed)
        }
    }
}

trait FormatValue {
    fn format_value(&self, use_float_format: bool) -> String;
}

impl FormatValue for f32 {
    fn format_value(&self, use_float_format: bool) -> String {
        if use_float_format {
            format_float(*self)
        } else {
            format!("{}", self)
        }
    }
}

impl FormatValue for f64 {
    fn format_value(&self, use_float_format: bool) -> String {
        if use_float_format {
            format_float(*self)
        } else {
            format!("{}", self)
        }
    }
}

impl FormatValue for bool {
    fn format_value(&self, _use_float_format: bool) -> String {
        format!("{}", self)
    }
}

macro_rules! impl_format_value_for_int {
    ($($t:ty),*) => {
        $(
            impl FormatValue for $t {
                fn format_value(&self, _use_float_format: bool) -> String {
                    format!("{}", self)
                }
            }
        )*
    };
}

impl_format_value_for_int!(u8, u16, u32, u64, i8, i16, i32, i64);

fn display_tensor_data<T>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    stride: usize,
    shape: &[usize],
    depth: usize,
    use_float_format: bool,
    column_width: Option<usize>,
) -> fmt::Result
where
    T: FormatValue,
{
    match shape.len() {
        0 => {
            write!(f, "{}", data[0].format_value(use_float_format))
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
                    .map(|val| val.format_value(use_float_format).len())
                    .max()
                    .unwrap_or(0)
            });

            for i in 0..show_start.min(len) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let formatted = data[i].format_value(use_float_format);
                write!(f, "{:width$}", formatted, width = width)?;
            }

            if len > show_start + show_end {
                write!(f, ", ... ")?;
            }

            if show_end > 0 && len > show_start + show_end {
                for i in (len - show_end)..len {
                    write!(f, ", ")?;
                    let formatted = data[i].format_value(use_float_format);
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
                            let formatted = data[idx].format_value(use_float_format);
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

fn debug_format_data<T>(data: &[T], use_float_format: bool) -> String
where
    T: FormatValue,
{
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
        result.push_str(&data[i].format_value(use_float_format));
    }

    if len > show_start + show_end {
        result.push_str(", ... ");
    }

    if show_end > 0 && len > show_start + show_end {
        for i in (len - show_end)..len {
            result.push_str(", ");
            result.push_str(&data[i].format_value(use_float_format));
        }
    }

    result.push(']');
    result
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Check for special constant tensors
        match self.id().0 {
            0 => return write!(f, "NULL_TENSOR"),
            _ => {},
        }

        if !self.is_const() && !self.is_storaged() {
            write!(f, "<lazy>")
        } else {
            match self.dtype() {
                DType::BF16 | DType::F16 => {
                    if let Ok(data) = self.try_to_flatten_vec::<f32>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F32 => {
                    if let Ok(data) = self.try_to_flatten_vec::<f32>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F64 => {
                    if let Ok(data) = self.try_to_flatten_vec::<f64>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::BOOL => {
                    if let Ok(data) = self.try_to_flatten_vec::<bool>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U8 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u8>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U16 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u16>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U32 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u32>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U64 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u64>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I8 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i8>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I16 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i16>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I32 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i32>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I64 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i64>() {
                        display_tensor_data(f, &data, self.size(), &self.shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
            }
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Check for special constant tensors
        match self.id().0 {
            0 => return write!(f, "NULL_TENSOR"),
            _ => {},
        }

        write!(
            f,
            "Tensor(id={}, device={}, dtype={}, shape=[",
            self.id().0,
            self.device().name(),
            self.dtype().as_str(),
        )?;
        let shape = self.shape();
        for (i, dim) in shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "], data=")?;

        if !self.is_const() && !self.is_storaged() {
            write!(f, "<lazy>")?;
        } else {
            match self.dtype() {
                DType::BF16 | DType::F16 => {
                    if let Ok(data) = self.try_to_flatten_vec::<f32>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F32 => {
                    if let Ok(data) = self.try_to_flatten_vec::<f32>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F64 => {
                    if let Ok(data) = self.try_to_flatten_vec::<f64>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::BOOL => {
                    if let Ok(data) = self.try_to_flatten_vec::<bool>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U8 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u8>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U16 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u16>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U32 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u32>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U64 => {
                    if let Ok(data) = self.try_to_flatten_vec::<u64>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I8 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i8>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I16 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i16>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I32 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i32>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I64 => {
                    if let Ok(data) = self.try_to_flatten_vec::<i64>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
            }
        }

        write!(f, ")")
    }
}
