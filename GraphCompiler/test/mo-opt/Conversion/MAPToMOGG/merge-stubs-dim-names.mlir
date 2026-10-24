// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg --mogg-conform-call-operands --mogg-outline-routines --mogg-to-mgp --mogg-canonicalize-funcs-for-merge --mogg-merge-identical-funcs --symbol-dce > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=STUBS < %t
// RUN: FileCheck %s --check-prefix=NONAME < %t

// Kernels that differ only in the names of their symbolic dims share one stub:
// CanonicalizeFuncsForMerge writes every dynamic dim of a stub's shape attrs as
// `dyn`, so MergeIdenticalFuncs sees the two stubs as identical.

// CHECK-LABEL: mgp.model @dyn_batch
// CHECK: mogg.call @[[STUB:__[A-Za-z0-9_]+]] %{{.*}} : !mogg.pointer, !mogg.pointer, !mogg.dim, !mogg.context
// CHECK-LABEL: mgp.model @dyn_seq
// CHECK: mogg.call @[[STUB]] %{{.*}} : !mogg.pointer, !mogg.pointer, !mogg.dim, !mogg.context

// STUBS-COUNT-1: mogg.func @
// STUBS-NOT: mogg.func @
// NONAME-NOT: ape[batch
// NONAME-NOT: ape[seq
mo.graph @dyn_batch(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}

mo.graph @dyn_seq(%a: !mo.tensor<[seq, 4], f32>) -> !mo.tensor<[seq, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[seq, 4], f32>) -> !mo.tensor<[seq, 4], f32>
  mo.output %c : !mo.tensor<[seq, 4], f32>
}
