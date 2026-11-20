// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-elementwise | FileCheck %s

// A same-shape, non-identity view (transpose of a square matrix) must not be
// absorbed by map-fuse-elementwise: ElementwiseFuser's compatibility check
// is shape-only, so without an explicit view check it would splice the
// transpose in as if it were a plain elementwise op, computing the
// transform but never using its result -- silently reading/writing at the
// wrong index. Fusing a view is ViewFuser's job. Repro from
// .fusion/.context.md.
mo.graph @no_fuse_transpose_of_square_matrix(
  %x: !mo.tensor<[3, 3], f32>,
  %y: !mo.tensor<[3, 3], f32>
) -> !mo.tensor<[3, 3], f32> {
  %sum = mo.add(%x, %y) : !mo.tensor<[3, 3], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %out = mo.transpose(%sum, %perm) : (
    !mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 3], f32>
  mo.output %out : !mo.tensor<[3, 3], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_transpose_of_square_matrix
// CHECK:       map.iter.elementwise<[3, 3]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.store
// CHECK:       map.iter.elementwise<[3, 3]>
// CHECK:       map.index.transpose
// CHECK:       map.store
// CHECK-NOT:   fusion_elementwise
