# Dropout Layer

The Dropout layer is a regularization technique that helps prevent neural networks from overfitting. It randomly sets a fraction of input units to zero during training, which helps prevent co-adaptation of neurons.

## Definition

```rust
pub struct Dropout {
    p: f32,
    state: LayerState,
}
```

## Constructor

```rust
pub fn new(p: f32) -> Result<Self>
```

Creates a new Dropout layer with the specified dropout probability.

**Parameters**:
- `p`: Probability of an element to be zeroed (between 0 and 1)

**Example**:
```rust
let dropout = Dropout::new(0.5)?; // 50% dropout probability
```

## Forward Pass

```rust
pub fn forward(&self, input: &Tensor) -> Result<Tensor>
```

Applies dropout to the input tensor.

**Parameters**:
- `input`: Input tensor of any shape

**Returns**: Output tensor of the same shape as input

**Example**:
```rust
// During training
let mut dropout = Dropout::new(0.5)?;
dropout.train(); // Activate training mode
let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
let y = dropout.forward(&x)?; // Some elements will be zeroed

// During evaluation
dropout.eval(); // Activate evaluation mode
let z = dropout.forward(&x)?; // No elements will be zeroed, same as input
```

## Behavior Differences in Training and Evaluation

Dropout behaves differently depending on the layer's state:

1. **Training Mode** (`is_training() == true`):
   - Randomly zeroes elements of the input tensor with probability `p`
   - Scales the remaining elements by a factor of `1/(1-p)` to maintain the expected sum
   - For example, with `p=0.5`, approximately half the elements will be zeroed, and the remaining elements will be multiplied by 2

2. **Evaluation Mode** (`is_training() == false`):
   - Identity function - returns the input unchanged
   - No elements are zeroed out

## Implementation Details

MaidenX's Dropout implementation includes:

1. A binary mask tensor that determines which elements to keep (1) or zero out (0)
2. A scaling factor of `1/(1-p)` applied to the kept elements to maintain the expected activation magnitude
3. Support for autograd to allow proper gradient flow during training

## Tips for Using Dropout

- Dropout is typically applied after activation functions
- Common dropout rates range from 0.1 to 0.5
- Higher dropout rates provide stronger regularization but may require longer training
- Always remember to call `layer.eval()` during inference/evaluation
- Dropout is often more effective in larger networks

## Example Usage in a Neural Network

```rust
// Define a simple neural network with dropout
let mut linear1 = Linear::new(784, 512, true)?;
let mut dropout1 = Dropout::new(0.2)?;
let mut linear2 = Linear::new(512, 10, true)?;

// Training loop
for _ in 0..num_epochs {
    // Set to training mode
    linear1.train();
    dropout1.train();
    linear2.train();
    
    let hidden = linear1.forward(&input)?;
    let hidden_dropped = dropout1.forward(&hidden)?; // Apply dropout
    let output = linear2.forward(&hidden_dropped)?;
    
    // Compute loss and update parameters
    // ...
}

// Evaluation
linear1.eval();
dropout1.eval(); // Important: disable dropout during evaluation
linear2.eval();

let hidden = linear1.forward(&test_input)?;
let hidden_dropped = dropout1.forward(&hidden)?; // No dropout is applied
let predictions = linear2.forward(&hidden_dropped)?;
```
