use criterion::{black_box, Criterion};
use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::Tensor;

// Constants for benchmark data sizes
const SIZES: [(usize, &str); 2] = [(1000, "small"), (10000, "medium")];

// Helper function for tensor creation and benchmarking
fn bench_op<F>(
    b: &mut criterion::Bencher,
    device: Device,
    dtype: DType,
    size: usize,
    data_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    op_fn: F,
) where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    // Generate initial data
    let raw_data: Vec<f32> = (0..size).map(|i| (i % 10) as f32 / 10.0).collect();
    let data = data_transform(raw_data);

    b.iter(|| {
        let mut x = Tensor::new(data.clone()).unwrap();
        x.with_device(device).unwrap();
        x.with_dtype(dtype).unwrap();
        black_box(op_fn(&x)).unwrap()
    })
}

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("unary/basic");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(3));
    group.sample_size(50);

    // Define operations with their data transformations and implementations
    let operations: Vec<(
        &str,
        Box<dyn Fn(Vec<f32>) -> Vec<f32>>,
        Box<dyn Fn(&Tensor) -> Result<Tensor>>,
    )> = vec![
        // Math operations
        (
            "abs",
            Box::new(|v| v.iter().map(|x| x - 0.5).collect()),
            Box::new(|x| x.abs()),
        ),
        ("neg", Box::new(|v| v), Box::new(|x| x.neg())),
        (
            "sign",
            Box::new(|v| v.iter().map(|x| x - 0.5).collect()),
            Box::new(|x| x.sign()),
        ),
        ("sqrt", Box::new(|v| v), Box::new(|x| x.sqrt())),
        ("pow", Box::new(|v| v), Box::new(|x| x.pow(2.0))),
        ("exp", Box::new(|v| v), Box::new(|x| x.exp())),
        (
            "log",
            Box::new(|v| v.iter().map(|x| x + 0.01).collect()),
            Box::new(|x| x.log()),
        ),
        // Trigonometric operations
        ("sin", Box::new(|v| v), Box::new(|x| x.sin())),
        ("cos", Box::new(|v| v), Box::new(|x| x.cos())),
        ("tan", Box::new(|v| v), Box::new(|x| x.tan())),
        // Neural network operations
        (
            "sigmoid",
            Box::new(|v| v.iter().map(|x| x * 10.0 - 5.0).collect()),
            Box::new(|x| x.sigmoid()),
        ),
        (
            "relu",
            Box::new(|v| v.iter().map(|x| x * 2.0 - 1.0).collect()),
            Box::new(|x| x.relu()),
        ),
        (
            "tanh",
            Box::new(|v| v.iter().map(|x| x * 2.0 - 1.0).collect()),
            Box::new(|x| x.tanh()),
        ),
        (
            "leaky_relu",
            Box::new(|v| v.iter().map(|x| x * 2.0 - 1.0).collect()),
            Box::new(|x| x.leaky_relu(0.01)),
        ),
        (
            "gelu",
            Box::new(|v| v.iter().map(|x| x * 2.0 - 1.0).collect()),
            Box::new(|x| x.gelu()),
        ),
        (
            "elu",
            Box::new(|v| v.iter().map(|x| x * 2.0 - 1.0).collect()),
            Box::new(|x| x.elu(1.0)),
        ),
    ];

    // Run benchmarks for CPU
    #[cfg(feature = "cpu")]
    {
        let device = Device::CPU;
        let dtype = DType::F32;

        for (op_name, data_transform, op_fn) in &operations {
            for &(size, size_name) in &SIZES {
                let bench_name = format!("{}/cpu/{}", op_name, size_name);

                group.bench_function(&bench_name, |b| bench_op(b, device, dtype, size, data_transform, op_fn));
            }
        }
    }

    // Run benchmarks for CUDA if enabled
    #[cfg(feature = "cuda")]
    {
        let device = Device::CUDA(0);
        let dtype = DType::F32;

        for (op_name, data_transform, op_fn) in &operations {
            for &(size, size_name) in &SIZES {
                let bench_name = format!("{}/cuda/{}", op_name, size_name);

                group.bench_function(&bench_name, |b| bench_op(b, device, dtype, size, data_transform, op_fn));
            }
        }
    }

    group.finish();
}
