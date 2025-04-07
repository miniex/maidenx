extern crate proc_macro;

use maidenx_macro_utils::MaidenXManifest;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Layer, attributes(layer))]
pub fn derive_layer(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let manifest = MaidenXManifest::default();
    let maidenx_core_path = manifest.get_path("maidenx_core");
    let maidenx_nn_path = manifest.get_path("maidenx_nn");
    let maidenx_tensor_path = manifest.get_path("maidenx_tensor");

    let num_inputs = ast
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("layer"))
        .and_then(|attr| {
            attr.parse_args::<syn::ExprAssign>().ok().and_then(|expr| {
                if let syn::Expr::Path(left) = *expr.left {
                    if left.path.is_ident("inputs") {
                        if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(n), .. }) = *expr.right {
                            n.base10_parse::<usize>().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        })
        .unwrap_or(1);

    let input_type = if num_inputs == 1 {
        quote!(&#maidenx_tensor_path::Tensor)
    } else {
        let tensor_refs = std::iter::repeat(quote!(&#maidenx_tensor_path::Tensor))
            .take(num_inputs)
            .collect::<Vec<_>>();
        quote!((#(#tensor_refs),*))
    };

    let layer_impl = quote! {
        impl #impl_generics #maidenx_nn_path::layer::Layer<#input_type>
            for #name #ty_generics #where_clause
        {
            fn forward(&self, input: #input_type)
                -> #maidenx_core_path::error::Result<#maidenx_tensor_path::Tensor> {
                self.forward(input)
            }
            fn parameters(&mut self) -> Vec<&mut #maidenx_tensor_path::Tensor> {
                self.parameters()
            }

            fn is_training(&self) -> bool {
                self.state.is_training()
            }
            fn train(&mut self)  {
                self.state.train();
            }
            fn eval(&mut self) {
                self.state.eval();
            }
        }
    };

    let save_load_impl = quote! {
        #[cfg(feature = "serde")]
        impl #impl_generics #name #ty_generics #where_clause {
            pub fn save<P: AsRef<std::path::Path>>(&self, path: P, format: &str) -> #maidenx_core_path::error::Result<()> {
                use std::io::Write;
                use std::path::{Path, PathBuf};

                let path_ref = path.as_ref();
                let mut path_with_ext = PathBuf::from(path_ref);

                match format {
                    "json" => {
                        if path_ref.extension().map_or(true, |ext| ext != "json") {
                            path_with_ext.set_extension("json");
                        }

                        let file = std::fs::File::create(&path_with_ext)
                            .map_err(|e| #maidenx_core_path::error::Error::External {
                                message: format!("Failed to create file: {}", e)
                            })?;
                        let mut writer = std::io::BufWriter::new(file);

                        let json = serde_json::to_string(self)
                            .map_err(|e| #maidenx_core_path::error::Error::SerializationError(
                                format!("Failed to serialize to JSON: {}", e)
                            ))?;

                        writer.write_all(json.as_bytes())
                            .map_err(|e| #maidenx_core_path::error::Error::External {
                                message: format!("Failed to write to file: {}", e)
                            })?;
                        writer.flush()
                            .map_err(|e| #maidenx_core_path::error::Error::External {
                                message: format!("Failed to flush buffer: {}", e)
                            })?;
                    },
                    "bytes" | "bin" => {
                        if path_ref.extension().map_or(true, |ext| ext != "bin") {
                            path_with_ext.set_extension("bin");
                        }

                        let file = std::fs::File::create(&path_with_ext)
                            .map_err(|e| #maidenx_core_path::error::Error::External {
                                message: format!("Failed to create file: {}", e)
                            })?;
                        let mut writer = std::io::BufWriter::new(file);

                        let config = bincode::config::legacy();
                        let bytes = bincode::serde::encode_to_vec(self, config)
                            .map_err(|e| #maidenx_core_path::error::Error::SerializationError(
                                format!("Failed to serialize to binary: {}", e)
                            ))?;

                        writer.write_all(&bytes)
                            .map_err(|e| #maidenx_core_path::error::Error::External {
                                message: format!("Failed to write to file: {}", e)
                            })?;
                        writer.flush()
                            .map_err(|e| #maidenx_core_path::error::Error::External {
                                message: format!("Failed to flush buffer: {}", e)
                            })?;
                    },
                    _ => {
                        return Err(#maidenx_core_path::error::Error::InvalidArgument(
                            format!("Invalid format '{}'. Use 'json' or 'bin'", format)
                        ));
                    }
                }

                Ok(())
            }

            pub fn load<P: AsRef<std::path::Path>>(path: P) -> #maidenx_core_path::error::Result<Self> {
                use std::path::{Path, PathBuf};

                let path_ref = path.as_ref();

                if !path_ref.exists() {
                    let mut json_path = PathBuf::from(path_ref);
                    json_path.set_extension("json");

                    let mut bin_path = PathBuf::from(path_ref);
                    bin_path.set_extension("bin");

                    if json_path.exists() {
                        return Self::load_json(&json_path);
                    } else if bin_path.exists() {
                        return Self::load_bin(&bin_path);
                    } else {
                        return Err(#maidenx_core_path::error::Error::External {
                            message: format!("File not found: {:?} (also tried with .json and .bin extensions)", path_ref)
                        });
                    }
                }

                if let Some(ext) = path_ref.extension().and_then(|ext| ext.to_str()) {
                    match ext {
                        "json" => Self::load_json(path_ref),
                        "bin" => Self::load_bin(path_ref),
                        _ => Err(#maidenx_core_path::error::Error::InvalidArgument(
                            format!("Unsupported file extension: {}. Expected .json or .bin", ext)
                        )),
                    }
                } else {
                    Err(#maidenx_core_path::error::Error::InvalidArgument(
                        "File has no extension. Please specify with .json or .bin extension".to_string()
                    ))
                }
            }

            fn load_json<P: AsRef<std::path::Path>>(path: P) -> #maidenx_core_path::error::Result<Self> {
                let content = std::fs::read_to_string(path)
                    .map_err(|e| #maidenx_core_path::error::Error::External {
                        message: format!("Failed to read JSON file: {}", e)
                    })?;

                serde_json::from_str(&content)
                    .map_err(|e| #maidenx_core_path::error::Error::DeserializationError(
                        format!("Failed to deserialize from JSON: {}", e)
                    ))
            }

            fn load_bin<P: AsRef<std::path::Path>>(path: P) -> #maidenx_core_path::error::Result<Self> {
                let bytes = std::fs::read(path)
                    .map_err(|e| #maidenx_core_path::error::Error::External {
                        message: format!("Failed to read binary file: {}", e)
                    })?;

                let config = bincode::config::legacy();
                bincode::serde::decode_from_slice(&bytes, config)
                    .map(|(value, _)| value)
                    .map_err(|e| #maidenx_core_path::error::Error::DeserializationError(
                        format!("Failed to deserialize from binary: {}", e)
                    ))
            }

            pub fn to_bytes(&self) -> #maidenx_core_path::error::Result<Vec<u8>> {
                let config = bincode::config::legacy();
                bincode::serde::encode_to_vec(self, config)
                    .map_err(|e| #maidenx_core_path::error::Error::SerializationError(
                        format!("Failed to serialize to bytes: {}", e)
                    ))
            }

            pub fn from_bytes(bytes: &[u8]) -> #maidenx_core_path::error::Result<Self> {
                let config = bincode::config::legacy();
                bincode::serde::decode_from_slice(bytes, config)
                    .map(|(value, _)| value)
                    .map_err(|e| #maidenx_core_path::error::Error::DeserializationError(
                        format!("Failed to deserialize from bytes: {}", e)
                    ))
            }

            pub fn to_json(&self) -> #maidenx_core_path::error::Result<String> {
                serde_json::to_string(self)
                    .map_err(|e| #maidenx_core_path::error::Error::SerializationError(
                        format!("Failed to serialize to JSON: {}", e)
                    ))
            }

            pub fn from_json(json: &str) -> #maidenx_core_path::error::Result<Self> {
                serde_json::from_str(json)
                    .map_err(|e| #maidenx_core_path::error::Error::DeserializationError(
                        format!("Failed to deserialize from JSON: {}", e)
                    ))
            }
        }
    };

    let expanded = quote! {
        #layer_impl
        #save_load_impl
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(Optimizer)]
pub fn derive_optimizer(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let manifest = MaidenXManifest::default();
    let maidenx_core_path = manifest.get_path("maidenx_core");
    let maidenx_nn_path = manifest.get_path("maidenx_nn");
    let maidenx_tensor_path = manifest.get_path("maidenx_tensor");

    let expanded = quote! {
        impl #impl_generics #maidenx_nn_path::optimizer::Optimizer for #name #ty_generics #where_clause {
            fn step(&mut self, parameters: &mut [&mut #maidenx_tensor_path::Tensor])
                -> #maidenx_core_path::error::Result<()> {
                self.step(parameters)
            }
            fn zero_grad(&mut self, parameters: &mut [&mut #maidenx_tensor_path::Tensor])
                -> #maidenx_core_path::error::Result<()> {
                self.zero_grad(parameters)
            }
            fn set_learning_rate(&mut self, learning_rate: impl Into<#maidenx_core_path::scalar::Scalar>) {
                self.set_learning_rate(learning_rate)
            }
        }
    };

    TokenStream::from(expanded)
}
