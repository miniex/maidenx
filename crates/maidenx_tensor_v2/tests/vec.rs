mod utils;

use maidenx_core::{device::auto_set_device, error::Result};
use maidenx_tensor_v2::Tensor;
use utils::test_both_modes;

#[test]
fn to_flatten_vec() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![1, 2, 3, 4]);
        assert_eq!(x.to_flatten_vec::<i32>(), [1, 2, 3, 4]);
        Ok(())
    })
}

#[test]
fn to_vec1d() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![1, 2]);
        assert_eq!(x.to_vec1d::<i32>(), [1, 2]);
        Ok(())
    })
}

#[test]
fn to_vec2d() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(x.to_vec2d::<i32>(), [[1, 2], [3, 4]]);
        Ok(())
    })
}

#[test]
fn to_vec3d() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]]);
        assert_eq!(x.to_vec3d::<i32>(), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        Ok(())
    })
}

#[test]
fn to_vec4d() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![
            vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]],
            vec![vec![vec![9, 10], vec![11, 12]], vec![vec![13, 14], vec![15, 16]]],
        ]);
        assert_eq!(
            x.to_vec4d::<i32>(),
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
            ]
        );
        Ok(())
    })
}

#[test]
fn to_vec5d() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![
            vec![
                vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]],
                vec![vec![vec![9, 10], vec![11, 12]], vec![vec![13, 14], vec![15, 16]]],
            ],
            vec![
                vec![vec![vec![17, 18], vec![19, 20]], vec![vec![21, 22], vec![23, 24]]],
                vec![vec![vec![25, 26], vec![27, 28]], vec![vec![29, 30], vec![31, 32]]],
            ],
        ]);
        assert_eq!(
            x.to_vec5d::<i32>(),
            [
                [
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
                ],
                [
                    [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                    [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]
                ]
            ]
        );
        Ok(())
    })
}

#[test]
fn to_vec6d() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let x = Tensor::new(vec![
            vec![
                vec![
                    vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]],
                    vec![vec![vec![9, 10], vec![11, 12]], vec![vec![13, 14], vec![15, 16]]],
                ],
                vec![
                    vec![vec![vec![17, 18], vec![19, 20]], vec![vec![21, 22], vec![23, 24]]],
                    vec![vec![vec![25, 26], vec![27, 28]], vec![vec![29, 30], vec![31, 32]]],
                ],
            ],
            vec![
                vec![
                    vec![vec![vec![33, 34], vec![35, 36]], vec![vec![37, 38], vec![39, 40]]],
                    vec![vec![vec![41, 42], vec![43, 44]], vec![vec![45, 46], vec![47, 48]]],
                ],
                vec![
                    vec![vec![vec![49, 50], vec![51, 52]], vec![vec![53, 54], vec![55, 56]]],
                    vec![vec![vec![57, 58], vec![59, 60]], vec![vec![61, 62], vec![63, 64]]],
                ],
            ],
        ]);
        assert_eq!(
            x.to_vec6d::<i32>(),
            [
                [
                    [
                        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
                    ],
                    [
                        [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                        [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]
                    ]
                ],
                [
                    [
                        [[[33, 34], [35, 36]], [[37, 38], [39, 40]]],
                        [[[41, 42], [43, 44]], [[45, 46], [47, 48]]]
                    ],
                    [
                        [[[49, 50], [51, 52]], [[53, 54], [55, 56]]],
                        [[[57, 58], [59, 60]], [[61, 62], [63, 64]]]
                    ]
                ]
            ]
        );
        Ok(())
    })
}
