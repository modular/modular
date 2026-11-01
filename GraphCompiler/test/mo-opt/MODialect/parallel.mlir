// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// A trivial parallel (body is just yield %arg) should be folded away,
// replacing all results with the corresponding inputs.

// CHECK-LABEL: mo.graph @fold_trivial_parallel
// CHECK-SAME: (%[[A:[^ ]*]]: !mo.tensor<[3], f32>, %[[B:[^ ]*]]: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
// CHECK-NEXT: mo.output %[[A]], %[[B]] : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
mo.graph @fold_trivial_parallel(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %res_bundle = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    mo.yield %arg : !mo.tensor<[3], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// A non-trivial parallel body should NOT be folded.

// CHECK-LABEL: mo.graph @keep_nontrivial_parallel
// CHECK: mo.parallel
mo.graph @keep_nontrivial_parallel(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %res_bundle = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32>
    mo.yield %r : !mo.tensor<[3], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// A parallel with no input bundles is syntactically valid as long as
// `launchDevices` is explicit.  This exercises the parser / printer /
// verifier only.  The body captures `%a` from outside so it is not
// lowerable in this form (lowering would expect a per-launch operand
// source) -- subsequent passes are not run here.

// CHECK-LABEL: mo.graph @parallel_no_input_bundles
// CHECK: mo.parallel () in ()
// CHECK-SAME: devices(gpu:0-1)
mo.graph @parallel_no_input_bundles(%a: !mo.tensor<[3], f32, gpu:0>) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) {
  %res_bundle = mo.parallel () in ()
      devices(gpu:0-1)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%a) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[3], f32, gpu:0-1>) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>
}
