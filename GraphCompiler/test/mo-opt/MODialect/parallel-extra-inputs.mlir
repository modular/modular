// RUN: mo-opt %s --canonicalize | FileCheck %s

// Verify that mo.parallel with tupled extra inputs parses and round-trips.
// The body receives two block arguments: the primary input and the extra input.

// CHECK-LABEL: mo.graph @parallel_with_extra_inputs
mo.graph @parallel_with_extra_inputs(
    %w0: !mo.tensor<[3], f32>,
    %w1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %in0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %in1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>> : !mo.tensor<[3], f32>

  %dt = mo.tensor.bundle(%in0, %in1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %de = mo.tensor.bundle(%w0, %w1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  // CHECK: mo.parallel
  // CHECK-SAME: in
  // CHECK-SAME: -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %res_bundle = mo.parallel (%arg, %extra) in (%dt : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %de : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %sum = mo.add(%arg, %extra) : !mo.tensor<[3], f32>
    mo.yield %sum : !mo.tensor<[3], f32>
  }

  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// Three input bundles: tensors + weights + bias.
// CHECK-LABEL: mo.graph @parallel_with_three_bundles
mo.graph @parallel_with_three_bundles(
    %in0: !mo.tensor<[3], f32>,
    %in1: !mo.tensor<[3], f32>,
    %w0: !mo.tensor<[3], f32>,
    %w1: !mo.tensor<[3], f32>,
    %b0: !mo.tensor<[3], f32>,
    %b1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %d_in = mo.tensor.bundle(%in0, %in1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_w = mo.tensor.bundle(%w0, %w1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_b = mo.tensor.bundle(%b0, %b1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  // CHECK: mo.parallel (%arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}) in
  %res_bundle = mo.parallel (%arg, %w, %b) in (%d_in : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_w : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_b : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %s = mo.add(%arg, %w) : !mo.tensor<[3], f32>
    %r = mo.add(%s, %b) : !mo.tensor<[3], f32>
    mo.yield %r : !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// Four input bundles: tensors + weights + bias + scale.
// CHECK-LABEL: mo.graph @parallel_with_four_bundles
mo.graph @parallel_with_four_bundles(
    %in0: !mo.tensor<[3], f32>,
    %in1: !mo.tensor<[3], f32>,
    %w0: !mo.tensor<[3], f32>,
    %w1: !mo.tensor<[3], f32>,
    %b0: !mo.tensor<[3], f32>,
    %b1: !mo.tensor<[3], f32>,
    %s0: !mo.tensor<[3], f32>,
    %s1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %d_in = mo.tensor.bundle(%in0, %in1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_w = mo.tensor.bundle(%w0, %w1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_b = mo.tensor.bundle(%b0, %b1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_s = mo.tensor.bundle(%s0, %s1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  // CHECK: mo.parallel (%arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}, %arg{{[0-9]+}}) in
  %res_bundle = mo.parallel (%arg, %w, %b, %sc) in (%d_in : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_w : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_b : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_s : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %t1 = mo.add(%arg, %w) : !mo.tensor<[3], f32>
    %t2 = mo.add(%t1, %b) : !mo.tensor<[3], f32>
    %t3 = mo.mul(%t2, %sc) : !mo.tensor<[3], f32>
    mo.yield %t3 : !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// Body yields two values, producing two output bundles.
// CHECK-LABEL: mo.graph @parallel_multi_yield
mo.graph @parallel_multi_yield(
    %a0: !mo.tensor<[3], f32>,
    %a1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %bundle = mo.tensor.bundle(%a0, %a1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  // CHECK: mo.parallel
  // CHECK-SAME: -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %out0, %out1 = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32>
    %n = mo.negative(%arg) : !mo.tensor<[3], f32>
    mo.yield %r, %n : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%out0) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  %n0, %n1 = mo.tensor.unbundle(%out1) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %r0, %r1, %n0, %n1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// Two output bundles with extra inputs (weights fed as second input bundle).
// CHECK-LABEL: mo.graph @parallel_multi_yield_with_extra
mo.graph @parallel_multi_yield_with_extra(
    %a0: !mo.tensor<[3], f32>,
    %a1: !mo.tensor<[3], f32>,
    %w0: !mo.tensor<[3], f32>,
    %w1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %d_in = mo.tensor.bundle(%a0, %a1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_w = mo.tensor.bundle(%w0, %w1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  // CHECK: mo.parallel
  // CHECK-SAME: -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %sum_bundle, %diff_bundle = mo.parallel (%arg, %w) in (%d_in : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_w : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %sum = mo.add(%arg, %w) : !mo.tensor<[3], f32>
    %diff = mo.sub(%arg, %w) : !mo.tensor<[3], f32>
    mo.yield %sum, %diff : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  }
  %s0, %s1 = mo.tensor.unbundle(%sum_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  %d0, %d1 = mo.tensor.unbundle(%diff_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %s0, %s1, %d0, %d1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// Three output bundles: exercises the yield-count == result-count contract
// with more than two outputs.
// CHECK-LABEL: mo.graph @parallel_three_outputs
mo.graph @parallel_three_outputs(
    %a0: !mo.tensor<[3], f32>,
    %a1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %bundle = mo.tensor.bundle(%a0, %a1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  // CHECK: mo.parallel
  // CHECK-SAME: -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %relu_b, %neg_b, %orig_b = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32>
    %n = mo.negative(%arg) : !mo.tensor<[3], f32>
    mo.yield %r, %n, %arg : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%relu_b) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  %n0, %n1 = mo.tensor.unbundle(%neg_b) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  %o0, %o1 = mo.tensor.unbundle(%orig_b) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %r0, %r1, %n0, %n1, %o0, %o1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
