use criterion::{black_box, Criterion};
use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::Tensor;

// Constants for benchmark data sizes
const SIZES: [(usize, &str); 3] = [(20, "small"), (200, "medium"), (1000, "large")];

// Helper function for tensor creation and benchmarking
fn bench_op<F>(
    b: &mut criterion::Bencher,
    device: Device, 
    dtype: DType,
    dims: &[usize],
    op_fn: F,
) 
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let data: Vec<f32> = (0..dims.iter().product::<usize>()).map(|i| i as f32).collect();
    
    b.iter(|| {
        let mut x = Tensor::new(data.clone()).unwrap().reshape(dims).unwrap();
        x.with_device(device).unwrap();
        x.with_dtype(dtype).unwrap();
        black_box(op_fn(&x)).unwrap()
    })
}

// Create tensor dimensions based on target size
fn create_dims(size: usize) -> Vec<usize> {
    if size <= 20 {
        // For small size, create a 4D tensor
        let dim = (size as f64).powf(0.25).ceil() as usize;
        vec![dim, dim, dim, dim]
    } else if size <= 200 {
        // For medium size, create a 3D tensor
        let dim = (size as f64).powf(1.0 / 3.0).ceil() as usize;
        vec![dim, dim, dim]
    } else {
        // For large size, create a 2D tensor
        let dim = (size as f64).sqrt().ceil() as usize;
        vec![dim, dim]
    }
}

pub fn basic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reduction/basic");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(3));
    group.sample_size(50);

    // Define operations with their implementations
    let operations: Vec<(&str, Box<dyn Fn(&Tensor) -> Result<Tensor>>)> = vec![
        ("sum_all", Box::new(|x| x.sum_all())),
        ("sum_dim_0", Box::new(|x| x.sum(0, false))),
        ("sum_dim_last", Box::new(|x| x.sum(x.shape().len() - 1, false))),
        ("sum_keepdim", Box::new(|x| x.sum(0, true))),
        ("mean_all", Box::new(|x| x.mean_all())),
        ("mean_dim_0", Box::new(|x| x.mean(0, false))),
        ("mean_dim_last", Box::new(|x| x.mean(x.shape().len() - 1, false))),
        ("max_all", Box::new(|x| x.max_all())),
        ("max_dim_0", Box::new(|x| x.max(0, false))),
        ("min_all", Box::new(|x| x.min_all())),
        ("min_dim_0", Box::new(|x| x.min(0, false))),
        ("norm_all", Box::new(|x| x.norm_all(2.0))),
        ("norm_dim_0", Box::new(|x| x.norm(2.0, 0, false))),
        ("var_dim_0", Box::new(|x| x.var(0, false, false))),
        ("std_dim_0", Box::new(|x| x.std(0, false, false))),
    ];

    // Run benchmarks for CPU
    #[cfg(feature = "cpu")]
    {
        let device = Device::CPU;
        let dtype = DType::F32;
        
        for (op_name, op_fn) in &operations {
            for &(size, size_name) in &SIZES {
                let dims = create_dims(size);
                let dims_str = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
                let bench_name = format!("{}/cpu/{}/{}", op_name, size_name, dims_str);
                
                group.bench_function(&bench_name, |b| {
                    bench_op(b, device, dtype, &dims, op_fn)
                });
            }
        }
    }

    // Run benchmarks for CUDA if enabled
    #[cfg(feature = "cuda")]
    {
        let device = Device::CUDA(0);
        let dtype = DType::F32;
        
        for (op_name, op_fn) in &operations {
            for &(size, size_name) in &SIZES {
                let dims = create_dims(size);
                let dims_str = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
                let bench_name = format!("{}/cuda/{}/{}", op_name, size_name, dims_str);
                
                group.bench_function(&bench_name, |b| {
                    bench_op(b, device, dtype, &dims, op_fn)
                });
            }
        }
    }

    group.finish();
}