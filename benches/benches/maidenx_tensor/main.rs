mod creation;
mod ops;

use criterion::criterion_main;

criterion_main!(ops::benches);
