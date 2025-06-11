// This module contains various tensor operations.

// ## Operations that create new tensors with new storage
mod binary;
mod matmul;
mod padding;
mod reduction;
mod unary;
// Indexing operations - all require tensor materialization before execution
mod indexing;

// ## Operations that create new tensors with shared storage
// These operations create new tensor objects but reuse the underlying storage.
mod transform;
