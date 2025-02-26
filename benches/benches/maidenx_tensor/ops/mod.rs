mod binary;

use criterion::criterion_group;

criterion_group!(benches, binary::basic);
