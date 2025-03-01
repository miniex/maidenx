mod binary;
mod matmul;
mod reduction;
mod transform;
mod unary;

mod total {
    use maidenx_core::error::Result;
    use maidenx_tensor::Tensor;

    #[test]
    fn test() -> Result<()> {
        let mut x = Tensor::new(vec![2.0])?;
        let mut y = Tensor::new(vec![3.0])?;
        x.with_grad()?;
        y.with_grad()?;

        let mut matrix1 = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let mut matrix2 = Tensor::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]])?;
        matrix1.with_grad()?;
        matrix2.with_grad()?;

        let z1 = x.mul(&y)?;
        let z2 = x.sub(&y)?;
        let z3 = x.add(&y)?;
        let z4 = x.pow(2)?.add(&y.pow(3)?)?;

        let sr = z1.add(&z2)?.add(&z3)?.add(&z4)?;
        let mr = matrix1.matmul(&matrix2)?.sum_all()?;

        let result = sr.add(&mr)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![51.0]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![9.0]);
        }
        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![29.0]);
        }

        Ok(())
    }
}
