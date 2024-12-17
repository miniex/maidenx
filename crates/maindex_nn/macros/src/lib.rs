extern crate proc_macro;

use maidenx_macro_utils::MaidenXManifest;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let manifest = MaidenXManifest::default();
    let maidenx_nn_path = manifest.get_path("maidenx_nn");
    let maidenx_tensor_path = manifest.get_path("maidenx_tensor");

    let num_inputs = ast
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("module"))
        .and_then(|attr| {
            attr.parse_args::<syn::ExprAssign>().ok().and_then(|expr| {
                if let syn::Expr::Path(left) = *expr.left {
                    if left.path.is_ident("inputs") {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Int(n),
                            ..
                        }) = *expr.right
                        {
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

    // 입력 타입 결정
    let input_type = if num_inputs == 1 {
        quote!(&#maidenx_tensor_path::Tensor)
    } else {
        let tensor_refs = std::iter::repeat(quote!(&#maidenx_tensor_path::Tensor))
            .take(num_inputs)
            .collect::<Vec<_>>();
        quote!((#(#tensor_refs),*))
    };

    let expanded = quote! {
        impl #impl_generics #maidenx_nn_path::module::Module<#input_type>
            for #name #ty_generics #where_clause
        {
            fn forward(&self, input: #input_type)
                -> #maidenx_nn_path::error::NnResult<#maidenx_tensor_path::Tensor> {
                self.forward(input)
            }
            fn parameters(&self) -> Vec<#maidenx_tensor_path::Tensor> {
                self.parameters()
            }
        }
    };
    TokenStream::from(expanded)
}
