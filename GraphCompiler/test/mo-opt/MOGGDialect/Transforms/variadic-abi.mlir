// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s

// End-to-end MAP-ABI emission of variadic operands. The `$variadic_abi`
// attribute `ConformCallOperands` stamps on `mogg.tuple.create` (from the
// callee's variadic parameter type) selects the aggregate
// `TupleCreateOp::emitMojo` builds, so emission never inspects the producing
// ops. Every kernel here takes its variadic operands as a trait-bound
// `TensorTuple`, fused or not.

// TENSOR TUPLE: mo.concat takes its fused variadic inputs as a trait-bound
// `TensorTuple`, so each element is its own fused load wrapper.
// CHECK: extensibility.TensorTuple[
mo.graph @tensor_tuple(%a: !mo.tensor<[2], si32>, %b: !mo.tensor<[3], si32>, %c: !mo.tensor<[4], si32>) -> !mo.tensor<[9], si32> {
  %result = mo.concat[0](%a, %b, %c) : (
    !mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[4], si32>
  ) -> !mo.tensor<[9], si32>
  mo.output %result : !mo.tensor<[9], si32>
}

// FUSED INPUT: the kernel's fused variadic inputs are a trait-bound
// `TensorTuple`, so each element is its own fused load wrapper.
// CHECK: extensibility.TensorTuple[
mo.graph @fused_input(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "variadic_add", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %c : !mo.tensor<[4], f32>
}

// FUSED OUTPUT: the kernel's store-epilogue-fused variadic outputs pack into a
// `TensorTuple` of fused store wrappers.
// CHECK: extensibility.TensorTuple[
mo.graph @fused_output(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_variadic_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// PLAIN: an unfused variadic input packs into a `TensorTuple` of dense
// arguments.
// CHECK: extensibility.TensorTuple[
mo.graph @plain(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_variadic", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %c : !mo.tensor<[4], f32>
}
