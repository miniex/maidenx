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

    let expanded = quote! {
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
