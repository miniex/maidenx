use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_cuda_path() -> String {
    const WINDOWS_CUDA_PATHS: &[&str] = &["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA", "C:/CUDA"];

    const UNIX_CUDA_PATHS: &[&str] = &["/opt/cuda", "/usr/local/cuda", "/usr/cuda"];

    // env
    if let Ok(path) = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")) {
        if PathBuf::from(&path).exists() {
            return path;
        }
    }

    // linux
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if let Ok(path) = String::from_utf8(output.stdout) {
            if let Some(cuda_path) = path.trim().strip_suffix("/bin/nvcc") {
                return cuda_path.to_string();
            }
        }
    }

    // default paths
    let default_paths = if cfg!(target_os = "windows") {
        WINDOWS_CUDA_PATHS.to_vec()
    } else {
        UNIX_CUDA_PATHS.to_vec()
    };

    for path in default_paths {
        if PathBuf::from(path).exists() {
            return path.to_string();
        }
    }

    "/usr/local/cuda".to_string()
}

fn get_compute_capability() -> String {
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if let Ok(cap) = String::from_utf8(output.stdout) {
            let cap = cap.trim();
            if !cap.is_empty() {
                return format!("sm_{}", cap.replace(".", ""));
            }
        }
    }
    "sm_90".to_string()
}

fn generate_clangd_config(cuda_path: &str, compute_capability: &str) -> String {
    format!(
        r#"CompileFlags:
  Remove: 
    - "-forward-unknown-to-host-compiler"
    - "-rdc=*"
    - "-Xcompiler*"
    - "--options-file"
    - "--generate-code*"
  Add: 
    - "-xcuda"
    - "-std=c++14"
    - "-I{}/include"
    - "-Ikernels"
    - "-Ikernels/ops"
    - "-Ikernels/nn"
    - "--cuda-gpu-arch={}"
  Compiler: clang
Index:
  Background: Build
Diagnostics:
  UnusedIncludes: None"#,
        cuda_path, compute_capability
    )
}

fn scan_cuda_files() {
    // Scan .cu files and ensure corresponding headers are accessible
    for entry in walkdir::WalkDir::new("kernels").into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().is_some_and(|ext| ext == "cu") {
            let cu_path = entry.path();
            let file_stem = cu_path.file_stem().unwrap().to_str().unwrap();

            // Check for corresponding .cuh in the same directory
            let local_cuh = cu_path.with_extension("cuh");

            // Check for corresponding .cuh in _headers directory
            let header_cuh = PathBuf::from("kernels/_headers").join(format!("{}.cuh", file_stem));

            println!("cargo:rerun-if-changed={}", cu_path.display());

            if local_cuh.exists() {
                println!("cargo:rerun-if-changed={}", local_cuh.display());
            }

            if header_cuh.exists() {
                println!("cargo:rerun-if-changed={}", header_cuh.display());
            }
        }
    }
}

fn main() {
    // Add walkdir as a build dependency
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=headers/");
    println!("cargo:rerun-if-changed=CMakeLists.txt");

    let cuda_path = find_cuda_path();
    let compute_capability = get_compute_capability();

    // Scan CUDA files and their headers
    scan_cuda_files();

    // Generate .clangd
    let clangd_path = PathBuf::from(".clangd");
    if !clangd_path.exists() {
        let clangd_content = generate_clangd_config(&cuda_path, &compute_capability);
        fs::write(&clangd_path, clangd_content).expect("Failed to write .clangd file");
    }

    let dst = cmake::Config::new(".")
        .define("CMAKE_BUILD_TYPE", "Release")
        .no_build_target(true)
        .build();

    // Search paths
    println!("cargo:rustc-link-search={}/build/lib", dst.display());
    println!("cargo:rustc-link-search={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search={}/lib64", cuda_path);
    println!("cargo:rustc-link-search={}/lib", cuda_path);

    // CUDA modules linking
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-lib=static=ops");
    println!("cargo:rustc-link-lib=static=nn");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    // CUDA runtime and core libraries
    println!("cargo:rustc-link-lib=cudart");
}
