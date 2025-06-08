// This module contains various tensor operations.

// ## Operations that create new tensors with new storage
mod binary;
mod indexing;
mod matmul;
mod padding;
mod reduction;
mod unary;

// ## Operations that create new tensors with shared storage
// These operations create new tensor objects but reuse the underlying storage.
mod transform;
