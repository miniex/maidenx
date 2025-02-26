pub mod layer;
pub mod optimizer;

pub mod layers;
pub mod losses;
pub mod optimizers;

pub use crate::{
    layers::linear::Linear,
    losses::{huber::Huber, mae::MAE, mse::MSE},
    optimizers::{adam::Adam, sgd::SGD},
};
