// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s --check-prefix=CHECK-MOGG
// RUN: mo-opt %s --canonicalize | FileCheck %s --check-prefix=CHECK-CANON

// Verify MO-to-MOGG preserves bundle/unbundle ops (not rejected as
// incompatible types), and that canonicalization folds the pair away.

// CHECK-MOGG-LABEL: mo.graph @bundle_fold
// CHECK-MOGG: mo.tensor.bundle
// CHECK-MOGG: mo.tensor.unbundle

// CHECK-CANON-LABEL: mo.graph @bundle_fold
// CHECK-CANON-NOT: mo.tensor.bundle
// CHECK-CANON-NOT: mo.tensor.unbundle
// CHECK-CANON: mo.relu
mo.graph @bundle_fold(
    %in0: !mo.tensor<[3], f32>,
    %in1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %b = mo.tensor.bundle(%in0, %in1)
      : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
     -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %r0, %r1 = mo.tensor.unbundle(%b)
      : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
     -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  %res0 = mo.relu(%r0) : !mo.tensor<[3], f32>
  %res1 = mo.relu(%r1) : !mo.tensor<[3], f32>
  mo.output %res0, %res1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
