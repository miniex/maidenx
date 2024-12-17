use crate::Tensor;
use std::fmt;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = match self.to_vec() {
            Ok(data) => data,
            Err(_) => return write!(f, "Tensor(Failed to fetch data)"),
        };
        let shape = self.shape();

        match shape.len() {
            1 => display_1d(f, &data, shape),
            2 => display_2d(f, &data, shape),
            3 => display_3d(f, &data, shape),
            4 => display_4d(f, &data, shape),
            _ => display_nd(f, &data, shape),
        }
    }
}

pub fn display_1d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    write!(f, "Tensor(shape=[{}],\\n  data=[", shape[0])?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{:.4}", val)?;
    }
    write!(f, "])")
}

pub fn display_2d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(f, "Tensor(shape=[{}, {}],", shape[0], shape[1])?;
    writeln!(f, "  data=[")?;
    for i in 0..shape[0] {
        write!(f, "    [")?;
        for j in 0..shape[1] {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", data[i * shape[1] + j])?;
        }
        writeln!(f, "],")?;
    }
    write!(f, "  ])")
}

pub fn display_3d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(
        f,
        "Tensor(shape=[{}, {}, {}],",
        shape[0], shape[1], shape[2]
    )?;
    writeln!(f, "  data=[")?;
    let plane_size = shape[1] * shape[2];

    for i in 0..shape[0] {
        writeln!(f, "    [")?;
        for j in 0..shape[1] {
            write!(f, "      [")?;
            for k in 0..shape[2] {
                if k > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", data[i * plane_size + j * shape[2] + k])?;
            }
            writeln!(f, "],")?;
        }
        writeln!(f, "    ],")?;
    }
    write!(f, "  ])")
}

pub fn display_4d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(
        f,
        "Tensor(shape=[{}, {}, {}, {}],",
        shape[0], shape[1], shape[2], shape[3]
    )?;
    writeln!(f, "  data=[")?;
    let volume_size = shape[1] * shape[2] * shape[3];
    let plane_size = shape[2] * shape[3];

    for w in 0..shape[0] {
        writeln!(f, "    [")?;
        for x in 0..shape[1] {
            writeln!(f, "      [")?;
            for y in 0..shape[2] {
                write!(f, "        [")?;
                for z in 0..shape[3] {
                    if z > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{:.4}",
                        data[w * volume_size + x * plane_size + y * shape[3] + z]
                    )?;
                }
                writeln!(f, "],")?;
            }
            writeln!(f, "      ],")?;
        }
        writeln!(f, "    ],")?;
    }
    write!(f, "  ])")
}

pub fn display_nd(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    write!(f, "Tensor(shape=[")?;
    for (i, &dim) in shape.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", dim)?;
    }
    writeln!(f, "],")?;
    write!(f, "  data=[")?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        if i >= 8 {
            write!(f, "...")?;
            break;
        }
        write!(f, "{:.4}", val)?;
    }
    write!(f, "])")
}
