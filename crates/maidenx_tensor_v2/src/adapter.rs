use half::{bf16, f16};
use maidenx_core::{dtype::DType, error::Result};

pub trait TensorAdapter: Sized {
    type Elem: Clone;

    fn to_flatten_vec(self) -> Result<Vec<Self::Elem>>;
    fn get_shape(&self) -> Vec<usize>;
    fn dtype(&self) -> DType;
}

macro_rules! impl_tensor_adapter {
    ($t:ty, $dtype:expr) => {
        // Scalar (Item Tensor)
        impl TensorAdapter for $t {
            type Elem = $t;

            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                Ok(vec![self])
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 1D Vector
        impl TensorAdapter for Vec<$t> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                Ok(self)
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![self.len()]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 2D Vector
        impl TensorAdapter for Vec<Vec<$t>> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend(row);
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
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
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in matrix {
                        flat.extend(row);
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
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
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
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
            fn get_shape(&self) -> Vec<usize> {
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
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
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
            fn get_shape(&self) -> Vec<usize> {
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
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
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
            fn get_shape(&self) -> Vec<usize> {
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

        // 1D Fixed-size Array Reference
        impl<'a, const N: usize> TensorAdapter for &'a [$t; N] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                Ok(self.to_vec())
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![N]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 2D Fixed-size Array Reference
        impl<'a, const M: usize, const N: usize> TensorAdapter for &'a [[$t; N]; M] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend_from_slice(row);
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![M, N]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 3D Fixed-size Array Reference
        impl<'a, const L: usize, const M: usize, const N: usize> TensorAdapter for &'a [[[$t; N]; M]; L] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in matrix {
                        flat.extend_from_slice(row);
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![L, M, N]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 4D Fixed-size Array Reference
        impl<'a, const K: usize, const L: usize, const M: usize, const N: usize> TensorAdapter
            for &'a [[[[$t; N]; M]; L]; K]
        {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor3d in self {
                    for matrix in tensor3d {
                        for row in matrix {
                            flat.extend_from_slice(row);
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![K, L, M, N]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 5D Fixed-size Array Reference
        impl<'a, const J: usize, const K: usize, const L: usize, const M: usize, const N: usize> TensorAdapter
            for &'a [[[[[$t; N]; M]; L]; K]; J]
        {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor4d in self {
                    for tensor3d in tensor4d {
                        for matrix in tensor3d {
                            for row in matrix {
                                flat.extend_from_slice(row);
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![J, K, L, M, N]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 6D Fixed-size Array Reference
        impl<'a, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize, const N: usize>
            TensorAdapter for &'a [[[[[[$t; N]; M]; L]; K]; J]; I]
        {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor5d in self {
                    for tensor4d in tensor5d {
                        for tensor3d in tensor4d {
                            for matrix in tensor3d {
                                for row in matrix {
                                    flat.extend_from_slice(row);
                                }
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                vec![I, J, K, L, M, N]
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 1D Slice of Slices
        impl<'a> TensorAdapter for &'a [&'a [$t]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend_from_slice(row);
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
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

        // 2D Slice of Slices
        impl<'a> TensorAdapter for &'a [&'a [&'a [$t]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in *matrix {
                        flat.extend_from_slice(row);
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len()]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 3D Slice of Slices
        impl<'a> TensorAdapter for &'a [&'a [&'a [&'a [$t]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor3d in self {
                    for matrix in *tensor3d {
                        for row in *matrix {
                            flat.extend_from_slice(row);
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len(), self[0][0][0].len()]
                }
            }
            fn dtype(&self) -> DType {
                $dtype
            }
        }

        // 4D Slice of Slices
        impl<'a> TensorAdapter for &'a [&'a [&'a [&'a [&'a [$t]]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor4d in self {
                    for tensor3d in *tensor4d {
                        for matrix in *tensor3d {
                            for row in *matrix {
                                flat.extend_from_slice(row);
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0, 0]
                } else if self[0][0][0].is_empty() {
                    vec![self.len(), self[0].len(), self[0][0].len(), 0, 0]
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

        // 5D Slice of Slices
        impl<'a> TensorAdapter for &'a [&'a [&'a [&'a [&'a [&'a [$t]]]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor5d in self {
                    for tensor4d in *tensor5d {
                        for tensor3d in *tensor4d {
                            for matrix in *tensor3d {
                                for row in *matrix {
                                    flat.extend_from_slice(row);
                                }
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0, 0, 0]
                } else if self[0][0][0].is_empty() {
                    vec![self.len(), self[0].len(), self[0][0].len(), 0, 0, 0]
                } else if self[0][0][0][0].is_empty() {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        0,
                        0,
                    ]
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

        // 6D Slice of Slices
        impl<'a> TensorAdapter for &'a [&'a [&'a [&'a [&'a [&'a [&'a [$t]]]]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Result<Vec<$t>> {
                let mut flat = Vec::new();
                for tensor6d in self {
                    for tensor5d in *tensor6d {
                        for tensor4d in *tensor5d {
                            for tensor3d in *tensor4d {
                                for matrix in *tensor3d {
                                    for row in *matrix {
                                        flat.extend_from_slice(row);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(flat)
            }
            fn get_shape(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0, 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0, 0, 0, 0]
                } else if self[0][0][0].is_empty() {
                    vec![self.len(), self[0].len(), self[0][0].len(), 0, 0, 0, 0]
                } else if self[0][0][0][0].is_empty() {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        0,
                        0,
                        0,
                    ]
                } else if self[0][0][0][0][0].is_empty() {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        0,
                        0,
                    ]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        self[0][0][0][0][0].len(),
                        self[0][0][0][0][0][0].len(),
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
