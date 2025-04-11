use half::{bf16, f16};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
};

pub trait TensorAdapter: Sized {
    type Elem: Clone;

    fn to_flat_vec(self) -> Result<Vec<Self::Elem>>;
    fn from_flat_vec(vec: Vec<Self::Elem>, shape: &[usize]) -> Result<Self>;
    fn to_shape(&self) -> Vec<usize>;
    fn dtype(&self) -> DType;
}

macro_rules! impl_tensor_adapter {
    ($t:ty, $dtype:expr) => {
        // Scalar (Item Tensor)
        impl TensorAdapter for $t {
            type Elem = $t;

            fn to_flat_vec(self) -> Result<Vec<$t>> {
                Ok(vec![self])
            }
            fn from_flat_vec(vec: Vec<$t>, _shape: &[usize]) -> Result<Self> {
                if vec.is_empty() {
                    return Err(Error::InvalidShape {
                        message: "Expected at least one value".into(),
                    });
                }
                Ok(vec[0])
            }
            fn to_shape(&self) -> Vec<usize> {
                vec![]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 1D Vector
        impl TensorAdapter for Vec<$t> {
            type Elem = $t;
            fn to_flat_vec(self) -> Result<Vec<$t>> {
                Ok(self)
            }
            fn from_flat_vec(vec: Vec<$t>, shape: &[usize]) -> Result<Self> {
                if shape.len() != 1 {
                    return Err(Error::InvalidShape {
                        message: "Expected 1D shape".into(),
                    });
                }
                if shape[0] != vec.len() {
                    return Err(Error::InvalidShape {
                        message: format!("Shape mismatch: expected {}, got {}", shape[0], vec.len()),
                    });
                }
                Ok(vec)
            }
            fn to_shape(&self) -> Vec<usize> {
                vec![self.len()]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 2D Vector
        impl TensorAdapter for Vec<Vec<$t>> {
            type Elem = $t;
            fn to_flat_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend(row);
                }
                Ok(flat)
            }
            fn from_flat_vec(vec: Vec<$t>, shape: &[usize]) -> Result<Self> {
                if shape.len() != 2 {
                    return Err(Error::InvalidShape {
                        message: "Expected 2D shape".into(),
                    });
                }
                let expected_len = shape[0] * shape[1];
                if expected_len != vec.len() {
                    return Err(Error::InvalidShape {
                        message: format!("Shape mismatch: expected {}, got {}", expected_len, vec.len()),
                    });
                }
                Ok(vec.chunks(shape[1]).map(|chunk| chunk.to_vec()).collect())
            }
            fn to_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0]
                } else {
                    vec![self.len(), self[0].len()]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 3D Vector
        impl TensorAdapter for Vec<Vec<Vec<$t>>> {
            type Elem = $t;
            fn to_flat_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in matrix {
                        flat.extend(row);
                    }
                }
                Ok(flat)
            }
            fn from_flat_vec(vec: Vec<$t>, shape: &[usize]) -> Result<Self> {
                if shape.len() != 3 {
                    return Err(Error::InvalidShape {
                        message: "Expected 3D shape".into(),
                    });
                }
                let expected_len = shape[0] * shape[1] * shape[2];
                if expected_len != vec.len() {
                    return Err(Error::InvalidShape {
                        message: format!("Shape mismatch: expected {}, got {}", expected_len, vec.len()),
                    });
                }
                Ok(vec
                    .chunks(shape[1] * shape[2])
                    .map(|chunk| {
                        chunk
                            .chunks(shape[2])
                            .map(|inner_chunk| inner_chunk.to_vec())
                            .collect()
                    })
                    .collect())
            }
            fn to_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len()]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 4D Vector
        impl TensorAdapter for Vec<Vec<Vec<Vec<$t>>>> {
            type Elem = $t;
            fn to_flat_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor3d in self {
                    for matrix in tensor3d {
                        for row in matrix {
                            flat.extend(row);
                        }
                    }
                }
                Ok(flat)
            }
            fn from_flat_vec(vec: Vec<$t>, shape: &[usize]) -> Result<Self> {
                if shape.len() != 4 {
                    return Err(Error::InvalidShape {
                        message: "Expected 4D shape".into(),
                    });
                }
                let expected_len = shape[0] * shape[1] * shape[2] * shape[3];
                if expected_len != vec.len() {
                    return Err(Error::InvalidShape {
                        message: format!("Shape mismatch: expected {}, got {}", expected_len, vec.len()),
                    });
                }
                Ok(vec
                    .chunks(shape[1] * shape[2] * shape[3])
                    .map(|chunk| {
                        chunk
                            .chunks(shape[2] * shape[3])
                            .map(|inner_chunk| {
                                inner_chunk
                                    .chunks(shape[3])
                                    .map(|innermost_chunk| innermost_chunk.to_vec())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect())
            }
            fn to_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len(), self[0][0][0].len()]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 5D Vector
        impl TensorAdapter for Vec<Vec<Vec<Vec<Vec<$t>>>>> {
            type Elem = $t;
            fn to_flat_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor4d in self {
                    for tensor3d in tensor4d {
                        for matrix in tensor3d {
                            for row in matrix {
                                flat.extend(row);
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn from_flat_vec(vec: Vec<$t>, shape: &[usize]) -> Result<Self> {
                if shape.len() != 5 {
                    return Err(Error::InvalidShape {
                        message: "Expected 5D shape".into(),
                    });
                }
                let expected_len = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
                if expected_len != vec.len() {
                    return Err(Error::InvalidShape {
                        message: format!("Shape mismatch: expected {}, got {}", expected_len, vec.len()),
                    });
                }
                Ok(vec
                    .chunks(shape[1] * shape[2] * shape[3] * shape[4])
                    .map(|chunk| {
                        chunk
                            .chunks(shape[2] * shape[3] * shape[4])
                            .map(|inner_chunk| {
                                inner_chunk
                                    .chunks(shape[3] * shape[4])
                                    .map(|inner2_chunk| {
                                        inner2_chunk
                                            .chunks(shape[4])
                                            .map(|innermost_chunk| innermost_chunk.to_vec())
                                            .collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect())
            }
            fn to_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                    ]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 6D Vector
        impl TensorAdapter for Vec<Vec<Vec<Vec<Vec<Vec<$t>>>>>> {
            type Elem = $t;
            fn to_flat_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor5d in self {
                    for tensor4d in tensor5d {
                        for tensor3d in tensor4d {
                            for matrix in tensor3d {
                                for row in matrix {
                                    flat.extend(row);
                                }
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn from_flat_vec(vec: Vec<$t>, shape: &[usize]) -> Result<Self> {
                if shape.len() != 6 {
                    return Err(Error::InvalidShape {
                        message: "Expected 6D shape".into(),
                    });
                }
                let expected_len = shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * shape[5];
                if expected_len != vec.len() {
                    return Err(Error::InvalidShape {
                        message: format!("Shape mismatch: expected {}, got {}", expected_len, vec.len()),
                    });
                }
                Ok(vec
                    .chunks(shape[1] * shape[2] * shape[3] * shape[4] * shape[5])
                    .map(|chunk| {
                        chunk
                            .chunks(shape[2] * shape[3] * shape[4] * shape[5])
                            .map(|inner_chunk| {
                                inner_chunk
                                    .chunks(shape[3] * shape[4] * shape[5])
                                    .map(|inner2_chunk| {
                                        inner2_chunk
                                            .chunks(shape[4] * shape[5])
                                            .map(|inner3_chunk| {
                                                inner3_chunk
                                                    .chunks(shape[5])
                                                    .map(|innermost_chunk| innermost_chunk.to_vec())
                                                    .collect()
                                            })
                                            .collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect())
            }
            fn to_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0, 0]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        self[0][0][0][0][0].len(),
                    ]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }
    };
}

impl_tensor_adapter!(bf16, DType::BF16);
impl_tensor_adapter!(f16, DType::F16);
impl_tensor_adapter!(f32, DType::F32);
impl_tensor_adapter!(f64, DType::F64);
impl_tensor_adapter!(bool, DType::BOOL);
impl_tensor_adapter!(u8, DType::U8);
impl_tensor_adapter!(u16, DType::U16);
impl_tensor_adapter!(u32, DType::U32);
impl_tensor_adapter!(u64, DType::U64);
impl_tensor_adapter!(i8, DType::I8);
impl_tensor_adapter!(i16, DType::I16);
impl_tensor_adapter!(i32, DType::I32);
impl_tensor_adapter!(i64, DType::I64);
