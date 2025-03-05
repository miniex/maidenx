use criterion::{black_box, Criterion};
use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for benchmark data sizes
const SMALL_SIZE: usize = 100;
const MEDIUM_SIZE: usize = 5000;
const LARGE_SIZE: usize = 10000;

// Helper function to setup tensor for benchmarks
fn setup_bench_tensor<T: Clone + 'static>(data: Vec<T>, device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = Tensor::new(data)?;
    tensor.with_device(device)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

// Helper function to generate data of specified size
fn generate_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| i as f32).collect()
}

// Run benchmark for specific operation, device and size
fn run_benchmark(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    op_name: &str,
    device: Device,
    device_name: &str,
    size: usize,
    size_name: &str,
    dtype: DType,
) {
    // Benchmark name format: "operation/device/size"
    let bench_name = format!("{}/{}/{}", op_name, device_name, size_name);

    match op_name {
        // Basic arithmetic operations
        "add" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.add(&y))?;
                    Ok(())
                })
            });
        }
        "sub" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.sub(&y))?;
                    Ok(())
                })
            });
        }
        "mul" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.mul(&y))?;
                    Ok(())
                })
            });
        }
        "div" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size).iter().map(|v| v + 2.0).collect(), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.div(&y))?;
                    Ok(())
                })
            });
        }

        // Inplace arithmetic operations
        "add_" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let mut x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.add_(&y))?;
                    Ok(())
                })
            });
        }
        "sub_" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let mut x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.sub_(&y))?;
                    Ok(())
                })
            });
        }
        "mul_" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let mut x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.mul_(&y))?;
                    Ok(())
                })
            });
        }
        "div_" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let mut x = setup_bench_tensor(generate_data(size).iter().map(|v| v + 2.0).collect(), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.div_(&y))?;
                    Ok(())
                })
            });
        }

        // Logical operations
        "logical_and" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.logical_and(&y))?;
                    Ok(())
                })
            });
        }
        "logical_or" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.logical_or(&y))?;
                    Ok(())
                })
            });
        }
        "logical_xor" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.logical_xor(&y))?;
                    Ok(())
                })
            });
        }

        // Comparison operations
        "eq" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.eq(&y))?;
                    Ok(())
                })
            });
        }
        "ne" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.ne(&y))?;
                    Ok(())
                })
            });
        }
        "lt" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.lt(&y))?;
                    Ok(())
                })
            });
        }
        "le" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.le(&y))?;
                    Ok(())
                })
            });
        }
        "gt" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.gt(&y))?;
                    Ok(())
                })
            });
        }
        "ge" => {
            group.bench_function(&bench_name, |b| {
                b.iter(|| -> Result<()> {
                    let x = setup_bench_tensor(generate_data(size), device, dtype)?;
                    let y = setup_bench_tensor(generate_data(size).iter().map(|v| v + 1.0).collect(), device, dtype)?;
                    black_box(x.ge(&y))?;
                    Ok(())
                })
            });
        }
        _ => panic!("Unknown operation: {}", op_name),
    }
}

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary/basic");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(3));
    group.sample_size(50);

    // Operation types to benchmark
    let operations = [
        // Basic arithmetic operations
        "add",
        "sub",
        "mul",
        "div",
        // Inplace operations
        "add_",
        "sub_",
        "mul_",
        "div_",
        // Logical operations
        "logical_and",
        "logical_or",
        "logical_xor",
        // Comparison operations
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ];

    // Data sizes for benchmarks
    let sizes = [(SMALL_SIZE, "small"), (MEDIUM_SIZE, "medium"), (LARGE_SIZE, "large")];

    // Run CPU benchmarks
    #[cfg(feature = "cpu")]
    {
        let cpu_device = Device::CPU;
        let dtype = DType::F32;
        for &op in &operations {
            for &(size, size_name) in &sizes {
                run_benchmark(&mut group, op, cpu_device, "cpu", size, size_name, dtype);
            }
        }
    }

    // Run CUDA benchmarks if the feature is enabled
    #[cfg(feature = "cuda")]
    {
        let cuda_device = Device::CUDA(0);
        let dtype = DType::F32;
        for &op in &operations {
            for &(size, size_name) in &sizes {
                run_benchmark(&mut group, op, cuda_device, "cuda", size, size_name, dtype);
            }
        }
    }

    group.finish();
}
