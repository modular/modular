// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-pipeline --map-to-mogg | FileCheck %s
//
// And, after `ConformCallOperands`, the read-only operand is recreated immutable
// (this is the fix; without it the call would carry a mutable slice the kernel's
// immutable parameter rejects at codegen).
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-pipeline --map-to-mogg --mogg-conform-call-operands | FileCheck %s --check-prefix=CONFORM

// `FuseMutableLoads` fuses a `mo.mutable.load` into an opaque kernel operand the
// kernel only reads -- here `map_lower_inplace_copy`'s immutable `src`. MAPToMOGG
// reconstructs every buffer operand as a mutable `mogg.tensor.create`
// (a `!mogg.buffer`), so both `dst` and the fused-read `src` come out mutable.
// `ConformCallOperands` then reconciles the read-only `src`: it accepts the
// mut->immut widening (OperandInference) and clones that operand's
// `mogg.tensor.create` immutable (`cloneWithMut`), yielding a `!mogg.tensor`
// whose codegen slice matches the kernel's `Input` parameter. `dst`, which the
// kernel genuinely writes, stays a mutable `!mogg.buffer`.

// Before conform: MAPToMOGG makes both operands mutable buffers.
// CHECK-LABEL: mo.graph @fused_mutable_load_into_immutable_input
// CHECK:     mogg.tensor.create({{.*}}) {{.*}}mut = true{{.*}}: !mogg.buffer
// CHECK:     mogg.tensor.create({{.*}}) {{.*}}mut = true{{.*}}: !mogg.buffer
// CHECK:     mogg.call.opaque @{{.*}}MapLowerInplaceCopy inputs(%{{.*}}, %{{.*}} : !mogg.buffer, !mogg.buffer)

// After conform: the read-only src is recreated immutable and the call uses it.
// CONFORM-LABEL: mo.graph @fused_mutable_load_into_immutable_input
// CONFORM:     mogg.tensor.create({{.*}}) {{.*}}mut = false{{.*}}: !mogg.tensor
// CONFORM:     mogg.call.opaque @{{.*}}MapLowerInplaceCopy inputs(%{{.*}}, %{{.*}} : !mogg.buffer, !mogg.tensor)
mo.graph @fused_mutable_load_into_immutable_input(%dst: !mo.buffer<[2, 2], f32>, %srcbuf: !mo.buffer<[2, 2], f32>) -> !mo.chain {
  %chain = mo.chain.create()
  %src, %loadchain = mo.mutable.load(%srcbuf, %chain) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outchain = mo.custom [symbol = "map_lower_inplace_copy", device = #M.device_ref<"cpu", 0>](%dst, %src, %chain) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
  mo.output %outchain : !mo.chain
}
