use metal::{CommandQueue, ComputePipelineState, Device, Library};
use std::sync::{Mutex, OnceLock};
use std::{collections::HashMap, path::Path};

pub fn initialize_ops() -> Result<(), KernelError> {
    initialize_metal()?;

    let context = get_metal_context()?;
    let libraries = context.libraries.lock().unwrap();

    if libraries.contains_key("ops") && libraries.contains_key("nn") {
        return Ok(());
    }

    drop(libraries);

    let metallib_ops_path = env!("MAIDENX_MPS_OPS_METALLIB_PATH");
    load_metal_library("ops", metallib_ops_path)?;

    #[cfg(feature = "nn")]
    {
        let metallib_nn_path = env!("MAIDENX_MPS_NN_METALLIB_PATH");
        load_metal_library("nn", metallib_nn_path)?;
    }

    Ok(())
}

pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
    libraries: Mutex<HashMap<String, Library>>,
}

static METAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

#[derive(Debug)]
pub enum KernelError {
    DeviceNotFound,
    LibraryNotFound(String),
    FunctionNotFound(String),
    PipelineCreationFailed(String),
    ContextNotInitialized,
    ExecutionFailed(String),
}

impl MetalContext {
    fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceNotFound)?;
        let command_queue = device.new_command_queue();
        Ok(Self {
            device,
            command_queue,
            pipelines: Mutex::new(HashMap::new()),
            libraries: Mutex::new(HashMap::new()),
        })
    }

    fn add_library(&self, name: &str, path: &str) -> Result<(), KernelError> {
        let library = self
            .device
            .new_library_with_file(Path::new(path))
            .map_err(|_| KernelError::LibraryNotFound(path.to_string()))?;
        let mut libraries = self.libraries.lock().unwrap();
        libraries.insert(name.to_string(), library);
        Ok(())
    }

    fn get_or_create_pipeline(&self, function_name: &str) -> Result<ComputePipelineState, KernelError> {
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(pipeline) = pipelines.get(function_name) {
                return Ok(pipeline.clone());
            }
        }
        let libraries = self.libraries.lock().unwrap();

        for library in libraries.values() {
            if let Ok(function) = library.get_function(function_name, None) {
                let pipeline = self
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| KernelError::PipelineCreationFailed(function_name.to_string()))?;

                let mut pipelines = self.pipelines.lock().unwrap();
                pipelines.insert(function_name.to_string(), pipeline.clone());
                return Ok(pipeline);
            }
        }
        Err(KernelError::FunctionNotFound(function_name.to_string()))
    }
}

pub fn initialize_metal() -> Result<(), KernelError> {
    if METAL_CONTEXT.get().is_none() {
        let context = MetalContext::new()?;
        let _ = METAL_CONTEXT.set(context);
    }
    Ok(())
}

pub fn get_metal_context() -> Result<&'static MetalContext, KernelError> {
    METAL_CONTEXT.get().ok_or(KernelError::ContextNotInitialized)
}

pub fn load_metal_library(name: &str, path: &str) -> Result<(), KernelError> {
    initialize_metal()?;

    let context = get_metal_context()?;
    context.add_library(name, path)
}

pub fn execute_function<F>(function_name: &str, setup_fn: F) -> Result<(), KernelError>
where
    F: FnOnce(ComputePipelineState, &CommandQueue, &Device) -> Result<(), KernelError>,
{
    initialize_metal()?;

    let context = get_metal_context()?;
    let pipeline = context.get_or_create_pipeline(function_name)?;

    setup_fn(pipeline, &context.command_queue, &context.device)
}
