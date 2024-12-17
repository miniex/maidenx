use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = [100, 1000, 10000, 100000]; // Batch size for benchmarking
    let dim1 = 64;
    let dim2 = 64;
    let output_dim = 32;
    let iterations = 20;

    for &size in &sizes {
        println!(
            "\nBenchmarking Bilinear Layer with batch size: {}, dim1: {}, dim2: {}, output_dim: {}, iterations: {}",
            size, dim1, dim2, output_dim, iterations
        );
        benchmark_bilinear_layer(size, dim1, dim2, output_dim, iterations)?;
    }

    Ok(())
}

fn benchmark_bilinear_layer(
    batch_size: usize,
    dim1: usize,
    dim2: usize,
    output_dim: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let input1_data: Vec<f32> = vec![1.0; batch_size * dim1];
    let input2_data: Vec<f32> = vec![1.0; batch_size * dim2];

    // CPU Benchmark
    {
        let _guard = DeviceGuard::new(Device::cpu())?;
        let mut cpu_times = Vec::new();

        for i in 0..iterations {
            let start = Instant::now();
            let input1 = Tensor::from_vec_with_device(
                input1_data.clone(),
                &[batch_size, dim1],
                &Device::cpu(),
            )?;
            let input2 = Tensor::from_vec_with_device(
                input2_data.clone(),
                &[batch_size, dim2],
                &Device::cpu(),
            )?;
            let bilinear = Bilinear::new_with_device(dim1, dim2, output_dim, None, Device::cpu())?;
            let output = bilinear.forward((&input1, &input2))?;
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
        let input1 = Tensor::from_vec_with_device(
            input1_data.clone(),
            &[batch_size, dim1],
            &Device::cuda(0),
        )?;
        let input2 = Tensor::from_vec_with_device(
            input2_data.clone(),
            &[batch_size, dim2],
            &Device::cuda(0),
        )?;
        let bilinear = Bilinear::new_with_device(dim1, dim2, output_dim, None, Device::cuda(0))?;
        let warmup = bilinear.forward((&input1, &input2))?;
        println!(
            "CUDA Result (first few elements): {:?}",
            &warmup.to_vec()?[..5]
        );

        for _ in 0..iterations {
            let start = Instant::now();
            let input1 = Tensor::from_vec_with_device(
                input1_data.clone(),
                &[batch_size, dim1],
                &Device::cuda(0),
            )?;
            let input2 = Tensor::from_vec_with_device(
                input2_data.clone(),
                &[batch_size, dim2],
                &Device::cuda(0),
            )?;
            let bilinear =
                Bilinear::new_with_device(dim1, dim2, output_dim, None, Device::cuda(0))?;
            let _output = bilinear.forward((&input1, &input2))?;
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
