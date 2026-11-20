// RUN: mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands | FileCheck %s

// ConformCallOperands wraps each trait-bound call operand in a
// `mogg.trait.create` carrying the distilled contract, so a raw
// `!mogg.tensor` (which reconstructs as a `ManagedTensorSlice`, conforming to
// neither `Tensor` nor `TileTensorable`) reaches the kernel as the
// conforming trait argument. The wrap is materialized in the IR here; the call
// op's own emit stays a dumb forward. `trait_negate` declares both its input
// and its output against `TileTensorable`, so both operands are wrapped --
// with the input carrying `role = input` and the output `role = output` -- and the
// execution device rides on the op for local alignment at emit.

// CHECK-LABEL: mo.graph @trait_negate_graph
// CHECK: %[[INTENSOR:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = false{{.*}} : !mogg.tensor
// CHECK: %[[OUTTENSOR:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = true{{.*}} : !mogg.tensor
// CHECK: %[[INARG:.*]] = mogg.trait.create(%[[INTENSOR]]) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[OUTARG:.*]] = mogg.trait.create(%[[OUTTENSOR]]) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}TraitNegate inputs(%[[INARG]] : !mogg.tensor) outputs(%[[OUTARG]] : !mogg.tensor) context(
mo.graph @trait_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %z = mo.custom [symbol = "trait_negate", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %z : !mo.tensor<[4], f32>
}
