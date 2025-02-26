#[macro_export]
macro_rules! bench {
    ($name:literal) => {
        concat!(module_path!(), "::", $name)
    };
}
