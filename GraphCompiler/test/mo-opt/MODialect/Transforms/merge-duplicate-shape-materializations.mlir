// RUN: mo-opt %s --mo-merge-duplicate-shape-materializations --cse --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: mo.graph @merge_params_test_case
mo.graph @merge_params_test_case<N1 -> O1>(
  %input: !mo.tensor<[], bf16>,
  %selector: !mo.tensor<[N1], bool>
) -> !mo.tensor<[O1], bf16> {

  %shape = mo.shape_of(%selector) : (!mo.tensor<[N1], bool>) -> !mo.tensor<[1], si32>

  // CHECK: mo.broadcast_to<() -> D1>
  // CHECK-NEXT: mo.rebind
  // CHECK-SAME: !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>

  %1 = mo.broadcast_to<() -> D1>(%input, %shape) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D1], bf16>
  %2 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
  %3 = mo.broadcast_to<() -> D2>(%input, %shape) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D2], bf16>
  %4 = mo.rebind(%3) : !mo.tensor<[D2], bf16> -> !mo.tensor<[N1], bf16>
  %res = mo.select(%selector, %2, %4) : (!mo.tensor<[N1], bool>, !mo.tensor<[N1], bf16>, !mo.tensor<[N1], bf16>) -> !mo.tensor<[N1], bf16>

  // CHECK: mo.output<N1>
  mo.output<N1> %res : !mo.tensor<[N1], bf16>
}

// -----

// CHECK-LABEL: mo.graph @llama_dupe_params
mo.graph @llama_dupe_params<N1, N2 -> O1, O2>(
  %input: !mo.tensor<[], bf16>,
  %selector: !mo.tensor<[N1, N2], bool>,
  %dummy: !mo.tensor<[N1, N1], bf16>
) -> !mo.tensor<[O1, O2], bf16> {

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, N1], bf16>) -> !mo.tensor<[2], si32>

  // CHECK: mo.broadcast_to<() -> D1, D2>
  // CHECK-NEXT: mo.rebind
  // CHECK-SAME: !mo.tensor<[D1, D2], bf16> -> !mo.tensor<[N1, D2], bf16>
  // CHECK-NEXT: mo.concat
  // CHECK-SAME: (!mo.tensor<[N1, D2], bf16>, !mo.tensor<[N1, N1], bf16>) -> !mo.tensor<[N1, N2], bf16>
  // CHECK-NEXT: kgen.param.declare assert1: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] out_dim must be equivalent to the sum of the input dims along the concat axis" : !kgen.string}, identical(:scalar<si64> add(D2, N1), N2))>

  %1 = mo.broadcast_to<() -> D1, D2>(%input, %shape) : (!mo.tensor<[], bf16>, !mo.tensor<[2], si32>) -> !mo.tensor<[D1, D2], bf16>
  %2 = mo.rebind(%1) : !mo.tensor<[D1, D2], bf16> -> !mo.tensor<[N1, D2], bf16>
  %3 = mo.concat[1] (%2, %dummy) : (!mo.tensor<[N1, D2], bf16>, !mo.tensor<[N1, N1], bf16>) -> !mo.tensor<[N1, N2], bf16>
  kgen.param.declare assert1: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] out_dim must be equivalent to the sum of the input dims along the concat axis" : !kgen.string}, identical(:scalar<si64> add(D2, N1), N2))>

  // CHECK-NEXT: kgen.param.declare assert2: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] out_dim must be equivalent to the sum of the input dims along the concat axis" : !kgen.string}, identical(:scalar<si64> add(D2, N1), N2))>
  %4 = mo.broadcast_to<() -> D3, D4>(%input, %shape) : (!mo.tensor<[], bf16>, !mo.tensor<[2], si32>) -> !mo.tensor<[D3, D4], bf16>
  %5 = mo.rebind(%4) : !mo.tensor<[D3, D4], bf16> -> !mo.tensor<[N1, D4], bf16>
  %6 = mo.concat[1] (%5, %dummy) : (!mo.tensor<[N1, D4], bf16>, !mo.tensor<[N1, N1], bf16>) -> !mo.tensor<[N1, N2], bf16>
  kgen.param.declare assert2: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] out_dim must be equivalent to the sum of the input dims along the concat axis" : !kgen.string}, identical(:scalar<si64> add(D4, N1), N2))>

  %res = mo.select(%selector, %3, %6) : (!mo.tensor<[N1, N2], bool>, !mo.tensor<[N1, N2], bf16>, !mo.tensor<[N1, N2], bf16>) -> !mo.tensor<[N1, N2], bf16>

  // CHECK: mo.output<N1, N2>
  mo.output<N1, N2> %res : !mo.tensor<[N1, N2], bf16>
}
