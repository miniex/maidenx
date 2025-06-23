use maidenx::tensor_v2::prelude::*;

fn main() {
    lazy!();

    let a = Tensor::new(vec![1.0, 2.0, 3.0]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0]);
    let c = Tensor::new(vec![1.0, 2.0, 3.0]);
    a.enable_grad();
    b.enable_grad();
    c.enable_grad();

    let d = a.add(&a.mul(&b)).add(&a.mul(&c)).add(&b.mul(&c));
    d.forward();
    d.backward();

    println!("d: {}", d);

    println!("a.grad: {}", a.grad());
    println!("b.grad: {}", b.grad());
    println!("c.grad: {}", c.grad());

    let e = a.add(&a.mul(&b).mul(&b)).add(&a.mul(&a)).add(&a.mul(&b).mul(&c));
    e.forward();
    e.backward();

    println!("e: {}", e);

    println!("a.grad: {}", a.grad());
    println!("b.grad: {}", b.grad());
    println!("c.grad: {}", c.grad());
}
