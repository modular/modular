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

// A strided slice over a dynamic dim, as emitted into the matmul's store
// functor: the select's runtime in-shape is built from the matmul's own extent
// `M` and its out-shape from the slice's `H`, both passed in as functor
// fields -- `Coord`s typed by each shape's static/dynamic split -- and handed
// to the primitive whole.

// CHECK:      def __{{.*}}, [[M:arg__[0-9]+]]: Int, [[H:arg__[0-9]+]]: Int,
// CHECK:      var [[IN_COORD:var__+[0-9]+]] = layout.Coord(Scalar[DType.int]([[M]]), layout.Idx[8])
// CHECK:      var [[OUT_COORD:var__+[0-9]+]] = layout.Coord(Scalar[DType.int]([[H]]), layout.Idx[8])
// CHECK:      var {{var__+[0-9]+}} = OutputStoreFunctor_{{[0-9]+}}[{{.*}}]({{.*}}, [[IN_COORD]], [[OUT_COORD]])

// CHECK:      struct OutputStoreFunctor_{{[0-9]+}}[spec__0: {{.*}}, spec__1: {{.*}}]
// CHECK:      var [[IN_FIELD:var__[0-9]+]]: layout.Coord[Int, layout.ComptimeInt[8]]
// CHECK-NEXT: var [[OUT_FIELD:var__[0-9]+]]: layout.Coord[Int, layout.ComptimeInt[8]]
// CHECK:      comptime [[STARTS:param___[0-9]+]] = layout.Coord(layout.Idx[1], layout.Idx[0])
// CHECK-NEXT: comptime [[STEPS:param___[0-9]+]] = layout.Coord(layout.Idx[2], layout.Idx[1])
// CHECK-NEXT: var {{var__+[0-9]+}} = builtin_primitives.primitives.mogg_index_slice_select[static_starts = [[STARTS]], static_steps = [[STEPS]]]({{var__+[0-9]+}}, self.[[IN_FIELD]], self.[[OUT_FIELD]])
mo.graph @matmul_slice_step_dynamic<M, H>(
  %a: !mo.tensor<[M, 4], f32>,
  %b: !mo.tensor<[4, 8], f32>,
  %stops: !mo.tensor<[2], si64>
) -> (!mo.tensor<[M, 8], f32>, !mo.tensor<[H, 8], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[M, 4], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[M, 8], f32>
  %slice = mo.slice(%mm, %starts, %stops, %steps) : (!mo.tensor<[M, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[H, 8], f32>
  mo.output %mm, %slice : !mo.tensor<[M, 8], f32>, !mo.tensor<[H, 8], f32>
}
