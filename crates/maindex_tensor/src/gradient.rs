use crate::{
    error::{TensorError, TensorResult},
    Tensor,
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, Weak},
};

#[allow(clippy::type_complexity)]
pub struct Node {
    pub grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
    pub inputs: Vec<Weak<Mutex<Node>>>,
    pub grad: Option<Tensor>,
}

impl Tensor {
    pub fn backward(&self) -> TensorResult<()> {
        if let Some(node) = &self.node {
            let mut grads = HashMap::new();
            let mut visited = HashSet::new();
            let mut stack = vec![];

            grads.insert(Arc::as_ptr(node), Tensor::ones(&self.shape)?);
            stack.push(Arc::clone(node));

            while let Some(current) = stack.pop() {
                let current_ptr = Arc::as_ptr(&current);

                if visited.contains(&current_ptr) {
                    continue;
                }

                visited.insert(current_ptr);

                if let Some(grad) = grads.get(&current_ptr).cloned() {
                    let mut current_guard = current.lock().unwrap();

                    current_guard.grad = match current_guard.grad.take() {
                        Some(existing_grad) => Some(existing_grad.add(&grad)?),
                        None => Some(grad.clone()),
                    };

                    if let Some(grad_fn) = &current_guard.grad_fn {
                        let input_grads = grad_fn(&grad);
                        let inputs = current_guard.inputs.clone();
                        drop(current_guard);

                        for (input_grad, input_weak) in input_grads.into_iter().zip(inputs.iter()) {
                            if let Some(input) = input_weak.upgrade() {
                                let input_ptr = Arc::as_ptr(&input);
                                match grads.entry(input_ptr) {
                                    std::collections::hash_map::Entry::Occupied(mut e) => {
                                        *e.get_mut() = e.get().add(&input_grad)?;
                                    }
                                    std::collections::hash_map::Entry::Vacant(e) => {
                                        e.insert(input_grad);
                                    }
                                }
                                stack.push(input);
                            }
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(TensorError::GradientError {
                reason: "No computation graph available for backward pass".to_string(),
            })
        }
    }

    pub fn grad(&self) -> TensorResult<Option<Tensor>> {
        if !self.requires_grad {
            return Err(TensorError::GradientError {
                reason: "This tensor does not require gradients.".to_string(),
            });
        }

        if let Some(node) = &self.node {
            let node_guard = node.lock().unwrap();
            Ok(node_guard.grad.clone())
        } else {
            Ok(None)
        }
    }
}
