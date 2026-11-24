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

// A strided window over a dynamic dim, as emitted into the matmul's store
// functor. The select's runtime in-shape is built from the matmul's extent `M`
// and its out-shape from the slice's `H`. Both pass into the functor as fields,
// `Coord`s typed by each shape's static/dynamic split, and go to the primitive
// whole. The runtime in-shape resolves the negative start.

// CHECK:      def __{{.*}}, [[M:arg__[0-9]+]]: Int, [[H:arg__[0-9]+]]: Int,
// CHECK:      var [[IN_COORD:var_+[0-9]+]] = layout.Coord(Scalar[DType.int]([[M]]), layout.Idx[16])
// CHECK:      var [[OUT_COORD:var_+[0-9]+]] = layout.Coord(Scalar[DType.int]([[H]]), layout.Idx[16])
// CHECK:      var {{var_+[0-9]+}} = OutputStoreFunctor_{{[0-9]+}}[{{.*}}]({{.*}}, [[IN_COORD]], [[OUT_COORD]])

// CHECK:      struct OutputStoreFunctor_{{[0-9]+}}
// CHECK:      var [[IN_FIELD:var_+[0-9]+]]: layout.Coord[Int, layout.ComptimeInt[16]]
// CHECK-NEXT: var [[OUT_FIELD:var_+[0-9]+]]: layout.Coord[Int, layout.ComptimeInt[16]]
// CHECK:      comptime [[STARTS:param_+[0-9]+]] = layout.Coord(layout.Idx[-8], layout.Idx[0])
// CHECK-NEXT: comptime [[STEPS:param_+[0-9]+]] = layout.Coord(layout.Idx[2], layout.Idx[1])
// CHECK-NEXT: var {{var_+[0-9]+}} = builtin_primitives.primitives.mogg_index_slice_select[static_starts = [[STARTS]], static_steps = [[STEPS]]]({{var_+[0-9]+}}, self.[[IN_FIELD]], self.[[OUT_FIELD]])
mo.graph @matmul_window_dynamic<M, H>(%p: !mo.tensor<[M, 8], f32>, %q: !mo.tensor<[8, 16], f32>, %b: !mo.tensor<[M, 16], f32>, %c: !mo.tensor<[M, 16], f32>, %stops: !mo.tensor<[2], si64>) -> (!mo.tensor<[M, 16], f32>, !mo.tensor<[M, 16], f32>, !mo.tensor<[H, 16], f32>) {
  %starts = mo.constant <#M.dense_array<-8, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%p, %q) : (!mo.tensor<[M, 8], f32>, !mo.tensor<[8, 16], f32>) -> !mo.tensor<[M, 16], f32>
  %x = mo.add(%mm, %b) : !mo.tensor<[M, 16], f32>
  %z = mo.add(%mm, %c) : !mo.tensor<[M, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[M, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[H, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[H, 16], f32>
  mo.output %x, %z, %y : !mo.tensor<[M, 16], f32>, !mo.tensor<[M, 16], f32>, !mo.tensor<[H, 16], f32>
}
