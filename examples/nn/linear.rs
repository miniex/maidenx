use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = [100, 1000, 10000, 100000]; // Batch size for benchmarking
    let input_dim = 128;
    let output_dim = 64;
    let iterations = 20;

    for &size in &sizes {
        println!(
            "\nBenchmarking Linear Layer with batch size: {}, input_dim: {}, output_dim: {}, iterations: {}",
            size, input_dim, output_dim, iterations
        );
        benchmark_linear_layer(size, input_dim, output_dim, iterations)?;
    }

    Ok(())
}

fn benchmark_linear_layer(
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_data: Vec<f32> = vec![1.0; batch_size * input_dim];

    // CPU Benchmark
    {
        let _guard = DeviceGuard::new(Device::cpu())?;
        let mut cpu_times = Vec::new();

        for i in 0..iterations {
            let start = Instant::now();
            let input = Tensor::from_vec_with_device(
                input_data.clone(),
                &[batch_size, input_dim],
                &Device::cpu(),
            )?;
            let mut linear = Linear::new(input_dim, output_dim, None, true)?;
            let output = linear.forward(&input)?;
            let elapsed = start.elapsed();
            cpu_times.push(elapsed);

            if i == 0 {
                println!(
                    "CPU Result (first few elements): {:?}",
                    &output.to_vec()?[..5]
                );
            }
        }

        let avg_cpu = cpu_times.iter().sum::<std::time::Duration>() / iterations as u32;
        let min_cpu = cpu_times.iter().min().unwrap();
        let max_cpu = cpu_times.iter().max().unwrap();
        println!(
            "CPU Times - Avg: {:?}, Min: {:?}, Max: {:?}",
            avg_cpu, min_cpu, max_cpu
        );
    }

    // CUDA Benchmark
    #[cfg(feature = "cuda")]
    {
        let _guard = DeviceGuard::new(Device::cuda(0))?;
        let mut gpu_times = Vec::new();

        // Warmup run
        let input = Tensor::from_vec_with_device(
            input_data.clone(),
            &[batch_size, input_dim],
            &Device::cuda(0),
        )?;
        let mut linear =
            Linear::new_with_device(input_dim, output_dim, None, true, Device::cuda(0))?;
        let warmup = linear.forward(&input)?;
        println!(
            "CUDA Result (first few elements): {:?}",
            &warmup.to_vec()?[..5]
        );

        for _ in 0..iterations {
            let start = Instant::now();
            let input = Tensor::from_vec_with_device(
                input_data.clone(),
                &[batch_size, input_dim],
                &Device::cuda(0),
            )?;
            let mut linear =
                Linear::new_with_device(input_dim, output_dim, None, true, Device::cuda(0))?;
            let _output = linear.forward(&input)?;
            gpu_times.push(start.elapsed());
        }

        let avg_gpu = gpu_times.iter().sum::<std::time::Duration>() / iterations as u32;
        let min_gpu = gpu_times.iter().min().unwrap();
        let max_gpu = gpu_times.iter().max().unwrap();
        println!(
            "CUDA Times - Avg: {:?}, Min: {:?}, Max: {:?}",
            avg_gpu, min_gpu, max_gpu
        );
    }

    Ok(())
}
