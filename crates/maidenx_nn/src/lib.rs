pub mod layer;
pub mod optimizer;

pub mod alias;
pub mod layers;
pub mod losses;
pub mod optimizers;

pub use crate::{
    alias::*,
    layers::linear::Linear,
    losses::{huber::Huber, mae::MAE, mse::MSE},
    optimizers::{adam::Adam, sgd::SGD},
};
