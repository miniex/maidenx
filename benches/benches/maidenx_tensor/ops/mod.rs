mod binary;
mod reduction;
mod unary;

use criterion::criterion_group;

criterion_group!(benches, binary::basic, unary::basic, reduction::basic);
