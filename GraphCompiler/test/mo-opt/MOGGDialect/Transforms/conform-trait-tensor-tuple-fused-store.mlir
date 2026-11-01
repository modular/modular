// A fused-output tuple only lowers with its epilogues fused, so both RUN lines
// run `--map-fuse-kernels`.
// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands | FileCheck %s

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

// `trait_tensor_tuple_fused_store_split` takes `outputs: TensorTuple[*Outs]`
// with `Outs: TypeList[Trait=Output & Fused & StoreAccess, ...]`. Each output's
// consumer fuses into its own tuple element: the element carries the tuple's
// shared fused-store contract, resolved through the call's result mapping, and
// the run packs into a `mogg.tuple.create` with the `tensor_tuple` ABI. The
// second result's position, 1, is past the kernel's one output argument, so a
// lookup by raw position would leave it without a contract.

// CHECK-LABEL: mo.graph @trait_tensor_tuple_fused_store_split_graph
// CHECK: %[[OUT0:.*]] = mogg.tensor.create.epilogue {{.*}}contract = <{role = output, fused = true, access = store, capabilities = none}>
// CHECK: %[[OUT1:.*]] = mogg.tensor.create.epilogue {{.*}}contract = <{role = output, fused = true, access = store, capabilities = none}>
// CHECK: %[[X:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[OUTS:.*]] = mogg.tuple.create(%[[OUT0]], %[[OUT1]]) variadic_abi = <tensor_tuple> : !mogg.tensor, !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}TraitTensorTupleFusedStoreSplit inputs(%[[X]] : !mogg.tensor) outputs(%[[OUTS]] : !mogg.tuple) {{.*}}hasEpilogueFusion = true

// Emit wraps each element in its own store-fusion wrapper, one per epilogue,
// and the tuple holds them.
// MOJO: var [[OUT0:[a-z0-9_]+]] = _MoggOutputStoreFusionTensor[DType.float32,
// MOJO: var [[OUT1:[a-z0-9_]+]] = _MoggOutputStoreFusionTensor[DType.int32,
// MOJO: var [[OUTS:[a-z0-9_]+]] = extensibility.TensorTuple[type_of([[OUT0]]), type_of([[OUT1]])]([[OUT0]], [[OUT1]])
// MOJO: TraitTensorTupleFusedStoreSplit.execute[target = "cpu"]([[OUTS]], {{[a-z0-9_]+}}, {{[a-z0-9_]+}})
mo.graph @trait_tensor_tuple_fused_store_split_graph(%x: !mo.tensor<[5], f32>, %c: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[3], f64>) {
  %a, %b = mo.custom [symbol = "trait_tensor_tuple_fused_store_split", device = #M.device_ref<"cpu", 0>] (%x) : (
    !mo.tensor<[5], f32>
  ) -> (!mo.tensor<[2], f32>, !mo.tensor<[3], si32>)
  %a2 = mo.add(%a, %c) : !mo.tensor<[2], f32>
  %b2 = mo.cast(%b) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], f64>
  mo.output %a2, %b2 : !mo.tensor<[2], f32>, !mo.tensor<[3], f64>
}
