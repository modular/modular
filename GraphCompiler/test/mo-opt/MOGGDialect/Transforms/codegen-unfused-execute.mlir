// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" | FileCheck %s --check-prefix=DECL
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

// `trait_unfused_execute` registers a fused and an unfused `execute`. Import
// records the fused signature plus the flag.
// DECL: mogg.kernel.decl @testing_kernels.testing_kernels.TraitUnfusedExecute registration = "trait_unfused_execute"
// DECL-SAME: supports_unfused_execute

// Mojo emission, one function per graph in graph order. With nothing fused,
// `ConformCallOperands` drops the pass-through epilogues and hands the kernel
// its outputs as plain dense arguments, so Mojo resolves the plain overload.
// CHECK-LABEL: def __custom__trait_unfused_execute{{[_0-9a-f]*}}(
// CHECK-NOT:   _MoggOutputStoreFusionTensor
// CHECK:       _MoggDenseTensorArg[{{[^,]+}}, {{[^,]+}}, {{[0-9]+}}, _MoggTensorIO.OUTPUT
// CHECK:       _MoggDenseTensorArg[{{[^,]+}}, {{[^,]+}}, {{[0-9]+}}, _MoggTensorIO.OUTPUT
// CHECK:       TraitUnfusedExecute.execute[target = "cpu"]
mo.graph @unfused(%x: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %z, %w = mo.custom [symbol = "trait_unfused_execute", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
  mo.output %z, %w : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// Fusing into `w` keeps the fused form for both outputs: `z` gets its
// pass-through store functor.
// CHECK-LABEL: def __custom__trait_unfused_execute{{[_0-9a-f]*}}(
// CHECK:       _MoggOutputStoreFusionTensor[
// CHECK:       _MoggOutputStoreFusionTensor[
// CHECK:       TraitUnfusedExecute.execute[target = "cpu"]
mo.graph @fused_other(%x: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %z, %w = mo.custom [symbol = "trait_unfused_execute", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
  %w2 = mo.add(%w, %x) : !mo.tensor<[4], f32>
  mo.output %z, %w2 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}
