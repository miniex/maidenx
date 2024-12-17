use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = [1000, 10000, 100000, 1000000, 10000000];
    let iterations = 25;

    for &size in &sizes {
        println!(
            "\nBenchmarking with size: {}, iterations: {}",
            size, iterations
        );
        benchmark_ops(size, iterations)?;
    }

    Ok(())
}

fn benchmark_ops(size: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    // CPU Benchmark
    {
        let _guard = DeviceGuard::new(Device::cpu())?;
        let mut cpu_times = Vec::new();

        for i in 0..iterations {
            let start = Instant::now();
            let a = Tensor::new(data_a.clone())?;
            let b = Tensor::new(data_b.clone())?;
            let c = a.add(&b)?;
            let elapsed = start.elapsed();
            cpu_times.push(elapsed);

            if i == 0 {
                println!("CPU Result (first few elements): {:?}", &c.to_vec()?[..5]);
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
        let _guard = DeviceGuard::new(Device::Cuda(0))?;
        let mut gpu_times = Vec::new();

        // Warmup run
        let a = Tensor::new(data_a.clone())?;
        let b = Tensor::new(data_b.clone())?;
        let warmup = a.add(&b)?;
        println!(
            "CUDA Result (first few elements): {:?}",
            &warmup.to_vec()?[..5]
        );

        for _ in 0..iterations {
            let start = Instant::now();
            let x = Tensor::new(data_a.clone())?;
            let y = Tensor::new(data_b.clone())?;
            let _z = x.add(&y)?;
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
