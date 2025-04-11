# Embedding Layer

The Embedding layer converts integer indices into dense vector representations. It's commonly used as the first layer in models that process text, categorical data, or any discrete input that needs to be mapped to a continuous vector space.

## Definition

```rust
pub struct Embedding {
    weight: Tensor,
    padding_idx: Option<usize>,
    max_norm: Option<f32>,
    norm_type: f32,
    scale_grad_by_freq: bool,
    state: LayerState,
}
```

## Constructor

```rust
pub fn new(num_embeddings: usize, embedding_dim: usize) -> Result<Self>
```

Creates a new Embedding layer with the specified dimensions.

**Parameters**:
- `num_embeddings`: Size of the vocabulary (number of possible indices)
- `embedding_dim`: Size of each embedding vector

**Example**:
```rust
let embedding = Embedding::new(10000, 300)?; // 10,000 words, 300-dimensional embeddings
```

For more control over the initialization and additional features, you can use the extended constructor:

```rust
pub fn new_with_spec(
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<usize>,
    max_norm: Option<f32>,
    norm_type: f32,
    scale_grad_by_freq: bool,
    device: Device,
    dtype: DType
) -> Result<Self>
```

**Additional Parameters**:
- `padding_idx`: If specified, entries at this index will be filled with zeros
- `max_norm`: If specified, embeddings will be normalized to have at most this norm
- `norm_type`: The p-norm to use for normalization (default: 2.0)
- `scale_grad_by_freq`: If true, gradients are scaled by the inverse of frequency of the words
- `device`: The device to place the layer's parameters on (CPU, CUDA, or MPS)
- `dtype`: The data type for the layer's parameters

**Example**:
```rust
let embedding = Embedding::new_with_spec(
    10000, 
    300, 
    Some(0),  // Index 0 is padding
    Some(5.0),  // Maximum norm of 5.0
    2.0,  // L2 norm
    true,  // Scale gradients by frequency
    Device::CUDA(0),
    DType::F32
)?;
```

## Forward Pass

```rust
pub fn forward(&self, input: &Tensor) -> Result<Tensor>
```

Retrieves embeddings for the given indices.

**Parameters**:
- `input`: Tensor of integer indices with any shape, dtype must be an integer type

**Returns**: Tensor of embeddings with shape `[*input.shape, embedding_dim]`

**Example**:
```rust
let embedding = Embedding::new(10, 5)?;
let indices = Tensor::new(vec![1, 3, 5, 7])?;
let embeddings = embedding.forward(&indices)?; // Shape: [4, 5]

// With batch dimension
let batch_indices = Tensor::new(vec![1, 3, 5, 7, 2, 4, 6, 8])?.reshape(&[2, 4])?;
let batch_embeddings = embedding.forward(&batch_indices)?; // Shape: [2, 4, 5]
```

## Parameter Access

```rust
pub fn weight(&self) -> &Tensor
```

Provides access to the embedding matrix.

**Example**:
```rust
let embedding = Embedding::new(10, 5)?;
let weights = embedding.weight(); // Shape: [10, 5]
```

Other accessor methods include:
- `num_embeddings()`
- `embedding_dim()`
- `padding_idx()`
- `max_norm()`
- `norm_type()`
- `scale_grad_by_freq()`

## Layer Implementation

The Embedding layer implements the `Layer` trait, providing methods for parameter collection and training state management:

```rust
pub fn parameters(&mut self) -> Vec<&mut Tensor>
```

Returns the embedding matrix as a trainable parameter.

## Common Use Cases

### Word Embeddings

```rust
// 10,000 words in vocabulary, 300-dim embeddings
let word_embedding = Embedding::new(10000, 300)?;

// Convert word indices to embeddings
let word_indices = Tensor::new(vec![42, 1337, 7, 42])?;
let word_vectors = word_embedding.forward(&word_indices)?; // Shape: [4, 300]
```

### One-Hot to Dense

```rust
// For 10 categorical values
let category_embedding = Embedding::new(10, 5)?;

// Convert category indices to embeddings
let categories = Tensor::new(vec![0, 3, 9, 2])?;
let category_vectors = category_embedding.forward(&categories)?; // Shape: [4, 5]
```

### With Padding

```rust
// Use index 0 as padding
let embedding = Embedding::new_with_spec(1000, 64, Some(0), None, 2.0, false, Device::CPU, DType::F32)?;

// Input with padding
let input = Tensor::new(vec![0, 1, 2, 0, 3, 0])?;
let embeddings = embedding.forward(&input)?; // Embeddings for index 0 will be all zeros
```

### With Maximum Norm

```rust
// Limit embedding norms to 1.0
let embedding = Embedding::new_with_spec(1000, 64, None, Some(1.0), 2.0, false, Device::CPU, DType::F32)?;

// All retrieved embeddings will have norm <= 1.0
let input = Tensor::new(vec![5, 10, 15])?;
let embeddings = embedding.forward(&input)?;
```

## Implementation Notes

- The embedding matrix is initialized with random values scaled by `1/sqrt(embedding_dim)`
- If `padding_idx` is specified, the corresponding embedding is filled with zeros
- If `max_norm` is specified, embeddings are normalized to have at most this norm
- If `scale_grad_by_freq` is true, gradients are scaled by the inverse frequency of the words in the mini-batch
