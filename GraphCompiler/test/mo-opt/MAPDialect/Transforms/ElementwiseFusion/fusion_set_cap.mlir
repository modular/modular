// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-elementwise=max-fusion-set-size=1000 | FileCheck %s --check-prefix=UNCAPPED
// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-elementwise=max-fusion-set-size=3 | FileCheck %s --check-prefix=CAPPED

// A residual chain of cheap (small-tensor) elementwise ops whose result has two
// readers, `%o1` and `%o2`. Input fusion duplicates a multi-use producer into
// every elementwise consumer, so the chain is recomputed in both, and because
// each step's operand is itself a fused producer the duplication compounds.
// Without a cap both outputs collapse into foreaches that recompute far more
// `add`s than the five in the source. The fusion-set cap bounds how many
// duplicated kernels land in one consumer, so the chain is split across several
// kernels instead.

mo.graph @residual_chain(%a: !mo.tensor<[64], f32, gpu:0>) -> (!mo.tensor<[64], f32, gpu:0>, !mo.tensor<[64], f32, gpu:0>) {
  %r0 = mo.exp(%a) : !mo.tensor<[64], f32, gpu:0>
  %d1 = mo.tanh(%r0) : !mo.tensor<[64], f32, gpu:0>
  %r1 = mo.add(%r0, %d1) : !mo.tensor<[64], f32, gpu:0>
  %d2 = mo.tanh(%r1) : !mo.tensor<[64], f32, gpu:0>
  %r2 = mo.add(%r1, %d2) : !mo.tensor<[64], f32, gpu:0>
  %d3 = mo.tanh(%r2) : !mo.tensor<[64], f32, gpu:0>
  %r3 = mo.add(%r2, %d3) : !mo.tensor<[64], f32, gpu:0>
  %d4 = mo.tanh(%r3) : !mo.tensor<[64], f32, gpu:0>
  %r4 = mo.add(%r3, %d4) : !mo.tensor<[64], f32, gpu:0>
  %d5 = mo.tanh(%r4) : !mo.tensor<[64], f32, gpu:0>
  %r5 = mo.add(%r4, %d5) : !mo.tensor<[64], f32, gpu:0>
  %o1 = mo.exp(%r5) : !mo.tensor<[64], f32, gpu:0>
  %o2 = mo.negative(%r5) : !mo.tensor<[64], f32, gpu:0>
  mo.output %o1, %o2 : !mo.tensor<[64], f32, gpu:0>, !mo.tensor<[64], f32, gpu:0>
}

// Uncapped: one foreach per output, with the residual adds duplicated well
// beyond the five in the source.
// UNCAPPED:          map.iter.elementwise<
// UNCAPPED:          map.iter.elementwise<
// UNCAPPED-COUNT-10: map.call.elementwise @{{.*}}.Add
// UNCAPPED-NOT:      map.iter.elementwise<

// Capped: the cap forces the chain across several kernels.
// CAPPED-COUNT-6: map.iter.elementwise<
