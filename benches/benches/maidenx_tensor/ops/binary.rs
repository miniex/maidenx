use criterion::{black_box, Criterion};
use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::Tensor;

// Constants for benchmark data sizes
const SIZES: [(usize, &str); 3] = [(100, "small"), (5000, "medium"), (10000, "large")];

// Helper function for tensor creation and benchmarking
fn bench_binary_op<F>(
    b: &mut criterion::Bencher,
    device: Device,
    dtype: DType,
    size: usize,
    // Functions to transform data for x and y tensors
    x_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    y_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    // The operation to benchmark
    op_fn: F,
) where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    // Generate base data
    let base_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let x_data = x_transform(base_data.clone());
    let y_data = y_transform(base_data);

    b.iter(|| {
        let mut x = Tensor::new(x_data.clone()).unwrap();
        x.with_device(device).unwrap();
        x.with_dtype(dtype).unwrap();

        let mut y = Tensor::new(y_data.clone()).unwrap();
        y.with_device(device).unwrap();
        y.with_dtype(dtype).unwrap();

        black_box(op_fn(&x, &y)).unwrap()
    })
}

// Helper function for in-place binary operations
fn bench_binary_op_inplace<F>(
    b: &mut criterion::Bencher,
    device: Device,
    dtype: DType,
    size: usize,
    x_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    y_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    op_fn: F,
) where
    F: Fn(&mut Tensor, &Tensor) -> Result<()>,
{
    // Generate base data
    let base_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let x_data = x_transform(base_data.clone());
    let y_data = y_transform(base_data);

    b.iter(|| {
        let mut x = Tensor::new(x_data.clone()).unwrap();
        x.with_device(device).unwrap();
        x.with_dtype(dtype).unwrap();

        let mut y = Tensor::new(y_data.clone()).unwrap();
        y.with_device(device).unwrap();
        y.with_dtype(dtype).unwrap();

        op_fn(&mut x, &y).unwrap()
    })
}

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary/basic");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(3));
    group.sample_size(50);

    // Define regular operations with their data transformations
    let operations: Vec<(
        &str,
        Box<dyn Fn(Vec<f32>) -> Vec<f32>>,
        Box<dyn Fn(Vec<f32>) -> Vec<f32>>,
        Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor>>,
    )> = vec![
        // Basic arithmetic operations
        (
            "add",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.add(y)),
        ),
        (
            "sub",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.sub(y)),
        ),
        (
            "mul",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.mul(y)),
        ),
        (
            "div",
            Box::new(|v| v.iter().map(|x| x + 2.0).collect()),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.div(y)),
        ),
        // Logical operations
        (
            "logical_and",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.logical_and(y)),
        ),
        (
            "logical_or",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.logical_or(y)),
        ),
        (
            "logical_xor",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.logical_xor(y)),
        ),
        // Comparison operations
        (
            "eq",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.eq(y)),
        ),
        (
            "ne",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.ne(y)),
        ),
        (
            "lt",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.lt(y)),
        ),
        (
            "le",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.le(y)),
        ),
        (
            "gt",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.gt(y)),
        ),
        (
            "ge",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.ge(y)),
        ),
    ];

    // Define in-place operations
    let inplace_operations: Vec<(
        &str,
        Box<dyn Fn(Vec<f32>) -> Vec<f32>>,
        Box<dyn Fn(Vec<f32>) -> Vec<f32>>,
        Box<dyn Fn(&mut Tensor, &Tensor) -> Result<()>>,
    )> = vec![
        (
            "add_",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.add_(y)),
        ),
        (
            "sub_",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.sub_(y)),
        ),
        (
            "mul_",
            Box::new(|v| v),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.mul_(y)),
        ),
        (
            "div_",
            Box::new(|v| v.iter().map(|x| x + 2.0).collect()),
            Box::new(|v| v.iter().map(|x| x + 1.0).collect()),
            Box::new(|x, y| x.div_(y)),
        ),
    ];

    // Run benchmarks for CPU
    #[cfg(feature = "cpu")]
    {
        let device = Device::CPU;
        let dtype = DType::F32;

        // Regular operations
        for (op_name, x_transform, y_transform, op_fn) in &operations {
            for &(size, size_name) in &SIZES {
                let bench_name = format!("{}/cpu/{}", op_name, size_name);

                group.bench_function(&bench_name, |b| {
                    bench_binary_op(b, device, dtype, size, x_transform, y_transform, op_fn)
                });
            }
        }

        // In-place operations
        for (op_name, x_transform, y_transform, op_fn) in &inplace_operations {
            for &(size, size_name) in &SIZES {
                let bench_name = format!("{}/cpu/{}", op_name, size_name);

                group.bench_function(&bench_name, |b| {
                    bench_binary_op_inplace(b, device, dtype, size, x_transform, y_transform, op_fn)
                });
            }
        }
    }

    // Run benchmarks for CUDA if enabled
    #[cfg(feature = "cuda")]
    {
        let device = Device::CUDA(0);
        let dtype = DType::F32;

        // Regular operations
        for (op_name, x_transform, y_transform, op_fn) in &operations {
            for &(size, size_name) in &SIZES {
                let bench_name = format!("{}/cuda/{}", op_name, size_name);

                group.bench_function(&bench_name, |b| {
                    bench_binary_op(b, device, dtype, size, x_transform, y_transform, op_fn)
                });
            }
        }

        // In-place operations
        for (op_name, x_transform, y_transform, op_fn) in &inplace_operations {
            for &(size, size_name) in &SIZES {
                let bench_name = format!("{}/cuda/{}", op_name, size_name);

                group.bench_function(&bench_name, |b| {
                    bench_binary_op_inplace(b, device, dtype, size, x_transform, y_transform, op_fn)
                });
            }
        }
    }

    group.finish();
}
