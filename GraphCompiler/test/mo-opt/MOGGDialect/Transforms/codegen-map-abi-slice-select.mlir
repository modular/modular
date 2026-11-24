// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s

// A strided window of a matmul's fused add, as emitted into the matmul's store
// functor. The select calls its primitive with the slice's static starts and
// steps as comptime params and its static in- and out-shapes as `Coord`s built
// in place. It emits its body, the relu and the window store at the
// primitive's index, only where the in-slice bit is set: on the `else` of
// `not` the bit. The whole stores of both adds follow at the functor's index.

// CHECK:      struct OutputStoreFunctor_{{[0-9]+}}
// CHECK:      var [[IDX_LIST:var_+[0-9]+]] = rebind[std.utils.IndexList[2]](std.utils.coord.coord_to_index_list(idx))
// CHECK-NEXT: var [[IN:var_+[0-9]+]] = layout.Coord(layout.Idx[8], layout.Idx[16])
// CHECK-NEXT: var [[OUT:var_+[0-9]+]] = layout.Coord(layout.Idx[4], layout.Idx[16])
// CHECK-NEXT: comptime [[STARTS:param_+[0-9]+]] = layout.Coord(layout.Idx[1], layout.Idx[0])
// CHECK-NEXT: comptime [[STEPS:param_+[0-9]+]] = layout.Coord(layout.Idx[2], layout.Idx[1])
// CHECK-NEXT: var [[INV:var_+[0-9]+]] = builtin_primitives.primitives.mogg_index_slice_select[static_starts = [[STARTS]], static_steps = [[STEPS]]]([[IDX_LIST]], [[IN]], [[OUT]])
// CHECK-NEXT: var [[J:var_+[0-9]+]] = layout.Coord([[INV]][0])
// CHECK-NEXT: var [[SEL:var_+[0-9]+]] = [[INV]][1]
// CHECK-NEXT: if not [[SEL]]:
// CHECK-NEXT:   pass
// CHECK-EMPTY:
// CHECK-NEXT: else:
// CHECK-NEXT:   var [[RELU:var_+[0-9]+]] = builtin_kernels.elementwise.ReLU.elementwise[dtype = DType.float32]({{var_+[0-9]+}})
// CHECK-NEXT:   managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}](std.utils.coord.coord_to_index_list([[J]])), rebind[{{.*}}]([[RELU]]))
// CHECK-EMPTY:
// CHECK-NEXT: managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}](std.utils.coord.coord_to_index_list(idx)), {{.*}}
// CHECK-NEXT: managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}](std.utils.coord.coord_to_index_list(idx)), {{.*}}
mo.graph @matmul_window(%p: !mo.tensor<[8, 8], f32>, %q: !mo.tensor<[8, 16], f32>, %b: !mo.tensor<[8, 16], f32>, %c: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<8, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%p, %q) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 16], f32>) -> !mo.tensor<[8, 16], f32>
  %x = mo.add(%mm, %b) : !mo.tensor<[8, 16], f32>
  %z = mo.add(%mm, %c) : !mo.tensor<[8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[8, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 16], f32>
  mo.output %x, %z, %y : !mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}
