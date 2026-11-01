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

// The concat+window epilogue, as emitted into its store functor: the concat
// store, then the select. The select calls its primitive with the slice's
// static starts (the negative one as written) and steps, its static shapes,
// and its runtime in- and out-shapes -- functor fields the kernel builds from
// the concat's own extents and the window's -- and emits its body, the window
// store at the primitive's index, only where its in-slice bit is set: on the
// `else` of `not` the bit, the branch sense of an early return. The runtime
// in-shape is what resolves the negative start.

// CHECK:      def __{{.*}}, [[B:arg__[0-9]+]]: Int, {{arg__[0-9]+}}: Int, [[T:arg__[0-9]+]]: Int,
// CHECK:      var [[OUT_COORD:var__+[0-9]+]] = layout.Coord(Scalar[DType.int]([[B]]), layout.Idx[8], layout.Idx[16])
// CHECK:      var [[IN_COORD:var__+[0-9]+]] = layout.Coord(Scalar[DType.int]([[B]]), Scalar[DType.int]([[T]]), layout.Idx[16])
// CHECK:      var [[IN_SHAPE:var__+[0-9]+]] = std.utils.coord.coord_to_index_list([[IN_COORD]])
// CHECK-NEXT: var [[OUT_SHAPE:var__+[0-9]+]] = std.utils.coord.coord_to_index_list([[OUT_COORD]])
// CHECK-NEXT: var {{var__+[0-9]+}} = EpilogueStoreFunctor_{{[0-9]+}}[{{.*}}]({{.*}}, [[IN_SHAPE]], [[OUT_SHAPE]])

// CHECK:      struct EpilogueStoreFunctor_{{[0-9]+}}[spec__0: {{.*}}, spec__1: {{.*}}]
// CHECK:      var [[IN_FIELD:var__[0-9]+]]: std.utils.IndexList[3]
// CHECK-NEXT: var [[OUT_FIELD:var__[0-9]+]]: std.utils.IndexList[3]
// CHECK:      managed_tensor_slice.simd_store_into_tensor_pointer[static_spec = Self.spec__0, simd_width = simd_width, element_alignment = element_alignment](self.{{var__[0-9]+}}, self.{{var__[0-9]+}}, self.{{var__[0-9]+}}, rebind[{{.*}}](std.utils.coord.coord_to_index_list([[IDX:var__+[0-9]+]])), rebind[{{.*}}]([[VAL:var__+[0-9]+]]))
// CHECK-NEXT: var [[IDX_LIST:var__+[0-9]+]] = rebind[std.utils.IndexList[3]](std.utils.coord.coord_to_index_list([[IDX]]))
// CHECK-NEXT: comptime [[STARTS:param___[0-9]+]] = layout.IntTuple(0, -8, 0)
// CHECK-NEXT: comptime [[STEPS:param___[0-9]+]] = layout.IntTuple(1, 1, 1)
// CHECK-NEXT: comptime [[IN_STATIC:param___[0-9]+]] = layout.IntTuple(-1, -1, 16)
// CHECK-NEXT: comptime [[OUT_STATIC:param___[0-9]+]] = layout.IntTuple(-1, 8, 16)
// CHECK-NEXT: var [[INV:var__+[0-9]+]] = builtin_primitives.primitives.mogg_index_slice_select[static_starts = [[STARTS]], static_steps = [[STEPS]], in_static_shape = [[IN_STATIC]], out_static_shape = [[OUT_STATIC]]]([[IDX_LIST]], self.[[IN_FIELD]], self.[[OUT_FIELD]])
// CHECK-NEXT: var [[J:var__+[0-9]+]] = layout.Coord([[INV]][0])
// CHECK-NEXT: var [[SEL:var__+[0-9]+]] = [[INV]][1]
// CHECK-NEXT: if not [[SEL]]:
// CHECK-NEXT:   pass
// CHECK-EMPTY:
// CHECK-NEXT: else:
// CHECK-NEXT:   managed_tensor_slice.simd_store_into_tensor_pointer[static_spec = Self.spec__1, simd_width = simd_width, element_alignment = element_alignment](self.{{var__[0-9]+}}, self.{{var__[0-9]+}}, self.{{var__[0-9]+}}, rebind[{{.*}}](std.utils.coord.coord_to_index_list([[J]])), rebind[{{.*}}]([[VAL]]))
mo.graph @concat_slice_window<B, S, T -> OB, OT>(
  %cache: !mo.tensor<[B, 8, 16], f32>,
  %new: !mo.tensor<[B, S, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  mo.output<B, T> %concat, %slice : !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}
