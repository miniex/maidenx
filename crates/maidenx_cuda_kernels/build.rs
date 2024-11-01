use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_cuda_path() -> String {
    // Linux
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if let Ok(path) = String::from_utf8(output.stdout) {
            if let Some(cuda_path) = path.trim().strip_suffix("/bin/nvcc") {
                return cuda_path.to_string();
            }
        }
    }

    // Windows
    for path in &[
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "C:/CUDA",
    ] {
        if PathBuf::from(path).exists() {
            return path.to_string();
        }
    }

    "/usr/local/cuda".to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=cuda-headers/");
    println!("cargo:rerun-if-changed=CMakeLists.txt");

    let cuda_path = find_cuda_path();
    let clangd_path = PathBuf::from(".clangd");

    if !clangd_path.exists() {
        let clangd_content = format!(
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
    - "-I../../cuda-headers"
    - "--cuda-gpu-arch=sm_75"
  Compiler: clang

Index:
  Background: Build

Diagnostics:
  UnusedIncludes: None"#,
            cuda_path
        );

        fs::write(".clangd", clangd_content).expect("Failed to write .clangd file");
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

    // CUDA modules linking - key changes here
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-lib=static=nn_ops");
    println!("cargo:rustc-link-lib=static=tensor_ops");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    // CUDA runtime linking
    println!("cargo:rustc-link-lib=cudart");
}
