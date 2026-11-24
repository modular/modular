// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mo-shape-inference --gc-pipeline=constant-fold=false -o /dev/null
// RUN: cat %t/*.mogg-pre-mgp.mlir | FileCheck %s

// A dynamic-permutation transpose fused into a dim-reducing consumer (reduce
// over the transposed last axis). The transposed view's reduced axis (D11) is
// a fresh runtime symbol that appears on no kernel-boundary tensor -- only on
// the intermediate view and the fused prologue's own shape. Kernel dynamic-dim
// collection walks every op in the body (not just the I/O signature), so D11
// still gets a `!mogg.dim` handle and the prologue's `mogg.shape.create`
// lowers instead of dereferencing a null dim. This is the compile-side
// regression guard for that collection; the end-to-end run stays excluded on a
// separate strided-view-through-prologue value bug.

// CHECK: mogg.tensor.create.transpose
// The rowwise iteration shape for the reduce input carries all four
// transposed dims, including the reduced axis D11 that lives only on the
// intermediate view.
// CHECK: %[[SHAPE:.*]] = mogg.shape.create({{.*}}) <shape = #mosh<ape[dyn, dyn, dyn, dyn]>{{.*}} : !mogg.dim<D8>, !mogg.dim<D9>, !mogg.dim<D10>, !mogg.dim<D11>
// CHECK-NEXT: mogg.with_trace {
// CHECK-NEXT: mogg.call.rowwise shape(%[[SHAPE]]) axis(3)

mo.graph @serving_default<D0, D1, D2, D3 -> D4, D5, D6>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[4], si32>) -> !mo.tensor<[D4, D5, D6, 1], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
  %0 = mo.transpose<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  %1 = mo.reduce.add(%0) : (!mo.tensor<[D8, D9, D10, D11], f32>) -> !mo.tensor<[D8, D9, D10, 1], f32>
  mo.output<D8, D9, D10> %1 : !mo.tensor<[D8, D9, D10, 1], f32>
}
