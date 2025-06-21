use maidenx::tensor_v2::prelude::*;

fn main() {
    lazy!();

    let x = Tensor::randn(&[3, 2]);
    let y = Tensor::randn(&[1, 2]);
    x.enable_grad();
    y.enable_grad();

    let z = x.div(&y);
    z.forward();
    println!("z: {:?}", z);

    x.grad().forward();
    y.grad().forward();

    println!("x.grad: {:?}", x.grad());
    println!("y.grad: {:?}", y.grad());
    println!("z.grad: {:?}", z.grad());
}
