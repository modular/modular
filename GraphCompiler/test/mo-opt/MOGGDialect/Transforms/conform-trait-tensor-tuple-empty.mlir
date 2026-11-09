// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands | FileCheck %s

// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false -o /dev/null
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=MOJO

// `trait_tensor_tuple_sum` takes `z`, `inputs: TensorTuple[*Ins]` and `bias`.
// With no operand for `inputs`, ConformCallOperands packs an empty tuple in its
// position, and emit passes an empty `TensorTuple`.

// CHECK-LABEL: mo.graph @trait_tensor_tuple_empty_graph
// CHECK: %[[INS:.*]] = mogg.tuple.create() variadic_abi = <tensor_tuple>
// CHECK: %[[BIAS:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}TraitTensorTupleSum inputs(%[[INS]], %[[BIAS]] : !mogg.tuple, !mogg.tensor)

// MOJO: var [[INS:[a-z0-9_]+]] = extensibility.TensorTuple()
// MOJO: TraitTensorTupleSum.execute({{[a-z0-9_]+}}, [[INS]], {{[a-z0-9_]+}}, {{[a-z0-9_]+}})
mo.graph @trait_tensor_tuple_empty_graph(%bias: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %z = mo.custom [symbol = "trait_tensor_tuple_sum", device = #M.device_ref<"cpu", 0>] (%bias) : (
    !mo.tensor<[2], f32>
  ) -> !mo.tensor<[2], f32>
  mo.output %z : !mo.tensor<[2], f32>
}

// `trait_tensor_tuple_sum_tail` takes `z`, `copies: TensorTuple[*Outs]`,
// `bias` and `inputs: TensorTuple[*Ins]`, so each tuple is last on its side.
// With neither bound, each side gets an empty tuple after its last operand.

// CHECK-LABEL: mo.graph @trait_tensor_tuple_empty_tail_graph
// CHECK: %[[BIAS:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[INS:.*]] = mogg.tuple.create() variadic_abi = <tensor_tuple>
// CHECK: %[[Z:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = output, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[OUTS:.*]] = mogg.tuple.create() variadic_abi = <tensor_tuple>
// CHECK: mogg.call.opaque @{{.*}}TraitTensorTupleSumTail inputs(%[[BIAS]], %[[INS]] : !mogg.tensor, !mogg.tuple) outputs(%[[Z]], %[[OUTS]] : !mogg.tensor, !mogg.tuple)

// CSE merges the two empty tuples, so emit passes one value in both positions.
// MOJO: var [[EMPTY:[a-z0-9_]+]] = extensibility.TensorTuple()
// MOJO: TraitTensorTupleSumTail.execute({{[a-z0-9_]+}}, [[EMPTY]], {{[a-z0-9_]+}}, [[EMPTY]], {{[a-z0-9_]+}})
mo.graph @trait_tensor_tuple_empty_tail_graph(%bias: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %z = mo.custom [symbol = "trait_tensor_tuple_sum_tail", device = #M.device_ref<"cpu", 0>] (%bias) : (
    !mo.tensor<[2], f32>
  ) -> !mo.tensor<[2], f32>
  mo.output %z : !mo.tensor<[2], f32>
}
