// RUN: mo-opt %s --canonicalize | FileCheck %s

// Test: tensor.bundle and tensor.unbundle round-trip.
// CHECK-LABEL: mo.graph @tensor_bundle_roundtrip
mo.graph @tensor_bundle_roundtrip(
    %t0: !mo.tensor<[3], f32, gpu:0>,
    %t1: !mo.tensor<[3], f32, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) {
  %b = mo.tensor.bundle(%t0, %t1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> (!mo.bundle<[3], f32, gpu:0-1>)
  // unbundle(bundle(x)) folds away:
  // CHECK-NOT: mo.tensor.bundle
  // CHECK-NOT: mo.tensor.unbundle
  %r0, %r1 = mo.tensor.unbundle(%b)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>
}

// Test: bundle(unbundle(%X)#0, unbundle(%X)#1) folds to %X.
// CHECK-LABEL: mo.graph @tensor_unbundle_rebundle
mo.graph @tensor_unbundle_rebundle(
    %b: !mo.bundle<[3], f32, gpu:0-1>
) -> (!mo.bundle<[3], f32, gpu:0-1>) {
  // Both ops should fold away, leaving only the argument passthrough:
  // CHECK-NOT: mo.tensor.unbundle
  // CHECK-NOT: mo.tensor.bundle
  %t0, %t1 = mo.tensor.unbundle(%b)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %b2 = mo.tensor.bundle(%t0, %t1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> (!mo.bundle<[3], f32, gpu:0-1>)
  mo.output %b2
      : !mo.bundle<[3], f32, gpu:0-1>
}

// Test: bundle that doesn't consume the full unbundle must NOT fold.
// CHECK-LABEL: mo.graph @tensor_rebundle_partial
mo.graph @tensor_rebundle_partial(
    %b: !mo.bundle<[3], f32, gpu:0-2>
) -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %t0, %t1, %t2 = mo.tensor.unbundle(%b)
      : (!mo.bundle<[3], f32, gpu:0-2>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
         !mo.tensor<[3], f32, gpu:2>)
  // Only two elements re-bundled; fold would lose element #2, so must NOT fire.
  // CHECK: mo.tensor.bundle
  %b2 = mo.tensor.bundle(%t0, %t1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> (!mo.bundle<[3], f32, gpu:0-1>)
  mo.output %b2
      : !mo.bundle<[3], f32, gpu:0-1>
}

// Test: bundle whose elements are from the same unbundle but out of order must
// NOT fold (changing element order is observable).
// CHECK-LABEL: mo.graph @tensor_rebundle_reordered
mo.graph @tensor_rebundle_reordered(
    %b: !mo.bundle<[3], f32, gpu:0-1>
) -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]>) {
  %t0, %t1 = mo.tensor.unbundle(%b)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  // Elements reordered (%t1, %t0); must NOT fold back to %b.
  // CHECK: mo.tensor.bundle
  %b2 = mo.tensor.bundle(%t1, %t0)
      : (!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>)
     -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]>)
  mo.output %b2
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]>
}

// Test: bundle whose elements come from DIFFERENT unbundle ops must NOT fold.
// CHECK-LABEL: mo.graph @tensor_rebundle_mixed_sources
mo.graph @tensor_rebundle_mixed_sources(
    %b1: !mo.bundle<[3], f32, gpu:0-1>,
    %b2: !mo.bundle<[3], f32, gpu:0-1>
) -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %a0, %a1 = mo.tensor.unbundle(%b1)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %c0, %c1 = mo.tensor.unbundle(%b2)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  // Mixed sources (%a0 from %b1, %c1 from %b2); must NOT fold.
  // CHECK: mo.tensor.bundle
  %out = mo.tensor.bundle(%a0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> (!mo.bundle<[3], f32, gpu:0-1>)
  mo.output %out
      : !mo.bundle<[3], f32, gpu:0-1>
}
