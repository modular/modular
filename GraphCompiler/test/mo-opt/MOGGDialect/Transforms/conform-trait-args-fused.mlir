// RUN: mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg | FileCheck %s

// A fused trait argument cannot be wrapped by `mogg.trait.create` downstream:
// its fusion functor is a local of the prologue/epilogue emit. So the contract
// is stamped onto the `mogg.tensor.create.prologue` / `.epilogue` producer at
// MAP lowering (where the fused-tensor -> call edge is still explicit), and emit
// reads that attr to wrap the functor in the conforming trait tensor rather than
// binding it onto a `ManagedTensorSlice`. Here we only assert the stamp on the
// prologue (the fused-input lane); the wrap + numerics, and the epilogue lane
// (which needs the accumulating-buffer pass), are covered by the e2e tests.

// `trait_fused_load_negate` declares its input `Some[Input & Fused & LoadAccess]`,
// so the `mo.add` producer fuses into it and the prologue carries a `load` input
// contract.
// CHECK-LABEL: mo.graph @fused_load
// CHECK: mogg.tensor.create.prologue
// CHECK-SAME: contract = <{role = input, fused = true, access = load
mo.graph @fused_load(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %p = mo.add(%x, %x) : !mo.tensor<[4], f32>
  %z = mo.custom [symbol = "trait_fused_load_negate", device = #M.device_ref<"cpu", 0>] (%p) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %z : !mo.tensor<[4], f32>
}
