use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let base_dir = Path::new(&manifest_dir);

    // Define directories
    let kernel_dir = base_dir.join("kernels");
    let output_dir = base_dir.join("build");
    let ops_dir = output_dir.join("ops");
    let nn_dir = output_dir.join("nn");
    let nn_activation_dir = nn_dir.join("activation");

    // Create output directories
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    fs::create_dir_all(&ops_dir).expect("Failed to create ops directory");
    fs::create_dir_all(&nn_dir).expect("Failed to create nn directory");
    fs::create_dir_all(&nn_activation_dir).expect("Failed to create nn/activation directory");

    // Define output files
    let ops_lib = output_dir.join("ops.metallib");
    let nn_lib = output_dir.join("nn.metallib");

    // Utility files
    let atomics = kernel_dir.join("atomics.metal");
    let atomics_air = output_dir.join("atomics.air");
    let utils = kernel_dir.join("metal_utils.metal");
    let utils_air = output_dir.join("metal_utils.air");

    // Compile utilities
    compile_metal_file(&atomics, &atomics_air, &kernel_dir);
    compile_metal_file(&utils, &utils_air, &kernel_dir);

    // Define and compile ops sources
    let ops_sources = vec![
        "ops/binary.metal",
        "ops/matmul.metal",
        "ops/padding.metal",
        "ops/reduction.metal",
        "ops/unary.metal",
    ];

    let mut ops_air_files = vec![atomics_air.clone(), utils_air.clone()];
    for source in &ops_sources {
        let source_path = kernel_dir.join(source);
        let filename = Path::new(source).file_stem().unwrap().to_str().unwrap();
        let air_path = ops_dir.join(format!("{}.air", filename));
        compile_metal_file(&source_path, &air_path, &kernel_dir);
        ops_air_files.push(air_path);
    }

    // Define and compile nn sources
    let nn_sources = vec!["nn/conv.metal"];

    let nn_activation_sources = vec!["nn/activation/softmax.metal"];

    let mut nn_air_files = vec![atomics_air, utils_air];
    for source in &nn_sources {
        let source_path = kernel_dir.join(source);
        let rel_path = Path::new(source);
        let _parent = rel_path.parent().unwrap().to_str().unwrap();
        let filename = rel_path.file_stem().unwrap().to_str().unwrap();
        let air_path = nn_dir.join(format!("{}.air", filename));
        compile_metal_file(&source_path, &air_path, &kernel_dir);
        nn_air_files.push(air_path);
    }

    for source in &nn_activation_sources {
        let source_path = kernel_dir.join(source);
        let rel_path = Path::new(source);
        let filename = rel_path.file_stem().unwrap().to_str().unwrap();
        let air_path = nn_activation_dir.join(format!("{}.air", filename));
        compile_metal_file(&source_path, &air_path, &kernel_dir);
        nn_air_files.push(air_path);
    }

    // Create metalllibs
    create_metallib(&ops_air_files, &ops_lib);
    create_metallib(&nn_air_files, &nn_lib);

    // Set cargo environment variables
    println!("cargo:rustc-env=MAIDENX_MPS_OPS_METALLIB_PATH={}", ops_lib.display());
    println!("cargo:rustc-env=MAIDENX_MPS_NN_METALLIB_PATH={}", nn_lib.display());

    // Make sure Cargo knows to rerun this script if any of the Metal files change
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=build/ops.metallib");
    println!("cargo:rerun-if-changed=build/nn.metallib");
}

fn compile_metal_file(source: &Path, output: &Path, include_dir: &Path) {
    println!("Compiling: {} -> {}", source.display(), output.display());

    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-I",
            include_dir.to_str().unwrap(),
            "-c",
            source.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute metal compiler");

    if !status.success() {
        panic!("Failed to compile {}", source.display());
    }
}

fn create_metallib(air_files: &[PathBuf], output: &Path) {
    println!("Creating metallib: {}", output.display());

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);

    for air_file in air_files {
        cmd.arg(air_file);
    }

    cmd.args(["-o", output.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to execute metallib command");

    if !status.success() {
        panic!("Failed to create metallib: {}", output.display());
    }
}

