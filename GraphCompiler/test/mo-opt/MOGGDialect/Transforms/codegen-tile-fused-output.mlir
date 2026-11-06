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
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=CALL
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=STORE
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=STORE-CALL
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=EW
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=EW-CALL

// The sm_100a tile `mo.matmul` takes a `_FusedComputeOutputTileTensor`, so the
// fused add becomes a `ComputeOutputFusionTile` whose `compute` stages the
// residual through the kernel's copier and returns the summed tile.
// CHECK: struct FusionStruct_{{[0-9]+}}[spec_0: {{.*}}, spec_1: {{.*}}](managed_tensor_slice.ComputeOutputFusionTile):
// CHECK:   var tile_0: layout.TileTensor[DType.float32, {{.*}}]
// CHECK:   var tile_1: layout.TileTensor[DType.float32, {{.*}}]
// CHECK:   def compute[dtype : DType, rank : Int, LayoutType : layout.TensorLayout, Copier : layout.tile_io.TileCopier](self, tile_coords: {{.*}}IndexList[rank], copier: Copier, val: layout.TileTensor[dtype, LayoutType, MutAnyOrigin]) -> layout.TileTensor[dtype, LayoutType, MutAnyOrigin]:
// CHECK:     var [[STAGE:.*]] = layout.TileTensor({{.*}}, val.layout)
// CHECK:     var [[SRC:.*]] = self.tile_1.tile[{{.*}}](layout.Coord(Int(tile_coords[0]), Int(tile_coords[1])))
// CHECK:     copier.copy([[STAGE]], [[SRC]].unsafe_address_space_cast[Copier.src_address_space]())
// CHECK:     var [[RHS:.*]] = layout.TileTensor({{.*}}AddressSpace.GENERIC{{.*}}, val.layout)
// CHECK:     var [[SUM:.*]] = builtin_kernels.elementwise.Add.elementwise(val, [[RHS]])
// CHECK:     return rebind[layout.TileTensor[dtype, LayoutType, MutAnyOrigin]]([[SUM]])
// CALL: var [[W:.*]] = {{.*}}._bind_to_fused_compute_output_tile(var__{{[0-9]+}})
// CALL: MatmulTileSM100.execute[{{.*}}has_epilogue_fusion = True{{.*}}]([[W]], {{.*}})

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "", tile_based_fusion = true>}} {
mo.graph @matmul_residual_add(%a: !mo.tensor<[256, 256], f32, gpu:0>, %b: !mo.tensor<[256, 256], f32, gpu:0>, %z: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.add(%0, %z) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}

// The `tile_rms_norm` test kernel takes a `_FusedOutputTileTensor`, so
// the fused mul becomes an `OutputFusionTile` whose `store` writes the product
// into its output capture with the kernel's store copier.
// STORE: struct FusionStruct_{{[0-9]+}}[{{.*}}](managed_tensor_slice.OutputFusionTile):
// STORE:   def store[dtype : DType, rank : Int, LayoutType : layout.TensorLayout, Copier : layout.tile_io.TileCopier, StoreCopier : layout.tile_io.TileCopier](self, tile_coords: {{.*}}IndexList[rank], copier: Copier, store_copier: StoreCopier, val: layout.TileTensor[dtype, LayoutType, MutAnyOrigin]):
// STORE:     copier.copy(
// STORE:     var [[PROD:.*]] = builtin_kernels.elementwise.Mul.elementwise(val, {{.*}})
// STORE:     var [[OUT:.*]] = self.tile_0.tile[{{.*}}](layout.Coord(Int(tile_coords[0]), Int(tile_coords[1])))
// STORE:     store_copier.copy([[OUT]].unsafe_address_space_cast[StoreCopier.dst_address_space](), [[PROD]].unsafe_address_space_cast[StoreCopier.src_address_space]())
// STORE-CALL: var [[W:.*]] = {{.*}}._bind_to_fused_output_tile(var__{{[0-9]+}})
// STORE-CALL: TileRMSNormStub.execute([[W]], {{.*}})

mo.graph @rms_norm_mul(%x: !mo.tensor<[256, 256], f32, gpu:0>, %gamma: !mo.tensor<[256], f32, gpu:0>, %scale: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.custom [symbol = "tile_rms_norm", device = #M.device_ref<"gpu", 0>](%x, %gamma) : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.mul(%0, %scale) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}

// A standalone elementwise chain becomes an `ElementwiseFusionTile` whose
// `compute` stages every input like the driver's `dst` and returns the tile
// `foreach_fusion_tile` stores.
// EW: struct FusionStruct_{{[0-9]+}}[spec_0: {{.*}}, spec_1: {{.*}}, spec_2: {{.*}}](managed_tensor_slice.ElementwiseFusionTile):
// EW:   def compute[dtype : DType, rank : Int, LayoutType : layout.TensorLayout, Copier : layout.tile_io.TileCopier](self, tile_coords: {{.*}}IndexList[rank], copier: Copier, dst: layout.TileTensor[dtype, LayoutType, MutAnyOrigin]) -> layout.TileTensor[dtype, LayoutType, MutAnyOrigin]:
// EW:     layout.TileTensor({{.*}}, dst.layout)
// EW:     copier.copy(
// EW:     var [[SUM:.*]] = builtin_kernels.elementwise.Add.elementwise(
// EW:     var [[PROD:.*]] = builtin_kernels.elementwise.Mul.elementwise([[SUM]],
// EW:     return rebind[layout.TileTensor[dtype, LayoutType, MutAnyOrigin]]([[PROD]])
// EW-CALL: builtin_primitives.primitives.foreach_fusion_tile(tensor = {{.*}}, elem = var__{{[0-9]+}}, ctx = {{.*}})

mo.graph @add_mul(%a: !mo.tensor<[256, 256], f32, gpu:0>, %b: !mo.tensor<[256, 256], f32, gpu:0>, %c: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.add(%a, %b) : !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.mul(%0, %c) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}
}
