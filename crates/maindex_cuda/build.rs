use std::path::PathBuf;
use std::process::Command;

fn find_cuda_path() -> String {
    const WINDOWS_CUDA_PATHS: &[&str] = &[
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "C:/CUDA",
    ];

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

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels/");
    println!("cargo:rerun-if-changed=cuda/headers/");
    println!("cargo:rerun-if-changed=cuda/CMakeLists.txt");

    let cuda_path = find_cuda_path();
    // let clangd_path = PathBuf::from("cuda/.clangd");

    let dst = cmake::Config::new("cuda")
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
    #[cfg(feature = "nn")]
    {
        println!("cargo:rustc-link-lib=static=nn_linear_layers");
        println!("cargo:rustc-link-lib=static=nn_non_linear_activations");
    }
    println!("cargo:rustc-link-lib=static=tensor_basic_ops");
    println!("cargo:rustc-link-lib=static=tensor_scalar_ops");
    println!("cargo:rustc-link-lib=static=tensor_shape_ops");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    // CUDA runtime linking
    println!("cargo:rustc-link-lib=cudart");
}
