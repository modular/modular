// RUN: mo-opt --mo-hoist-constant-subgraphs-into-parent --split-input-file %s | FileCheck %s


// Test case: Single callsite with constants - NO hoisting (requires multiple callsites)
// CHECK-LABEL: module
module {

// Graph should remain unchanged because there's only one callsite
// CHECK-LABEL: mo.graph @single_callsite
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[ARG1:.+]]: !mo.tensor<[4], f32>, %[[ARG2:.+]]: !mo.tensor<[4], f32>)
mo.graph @single_callsite(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>, %arg2: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) isSubgraph {
  // These ops would be hoistable if there were multiple callsites with same invariant args
  // CHECK: %[[MUL_OUT:.+]] = mo.mul(%[[ARG1]], %[[ARG2]])
  %0 = mo.mul(%arg1, %arg2) : !mo.tensor<[4], f32>
  // CHECK: %[[RELU_OUT:.+]] = mo.relu(%[[MUL_OUT]])
  %1 = mo.relu(%0) : !mo.tensor<[4], f32>

  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[RELU_OUT]])
  %2 = mo.add(%arg0, %1) : !mo.tensor<[4], f32>

  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %2 : !mo.tensor<[4], f32>
}

// Caller should remain unchanged, no hoisting happens
// CHECK-LABEL: mo.graph @caller_single_callsite
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_single_callsite(%arg0: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst1 = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  // CHECK: %[[CONST2:.+]] = mo.constant.external [name = "cst2", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst2 = mo.constant.external [name = "cst2", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>

  // CHECK: %[[CALL_RESULT:.+]] = mo.call @single_callsite(%[[CALLER_ARG0]], %[[CONST1]], %[[CONST2]])
  %res = mo.call @single_callsite(%arg0, %cst1, %cst2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  // CHECK: mo.output %[[CALL_RESULT]]
  mo.output %res : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Multiple callsites with same invariant argument - hoisting should happen
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @callsite_invariant
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG:.+]]: !mo.tensor<[4], f32>)
mo.graph @callsite_invariant(%arg0: !mo.tensor<[4], f32>, %arg1_invariant: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) isSubgraph {
  // This op depends only on callsite-invariant arg1 and should be hoisted
  %0 = mo.relu(%arg1_invariant) : !mo.tensor<[4], f32>

  // This op depends on non-invariant arg0 and the hoisted boundary value
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG]])
  %1 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>

  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %1 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_callsite_invariant
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_callsite_invariant(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST:.+]] = mo.constant.external [name = "cst", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst = mo.constant.external [name = "cst", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>

  // Hoisted ops for both calls (same invariant argument %cst)
  // CHECK: %[[HOISTED_RELU1:.+]] = mo.relu(%[[CONST]])
  // CHECK: %[[CALL1_RESULT:.+]] = mo.call @callsite_invariant(%[[CALLER_ARG0]], %[[HOISTED_RELU1]])
  %res1 = mo.call @callsite_invariant(%arg0, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[HOISTED_RELU2:.+]] = mo.relu(%[[CONST]])
  // CHECK: %[[CALL2_RESULT:.+]] = mo.call @callsite_invariant(%[[CALLER_ARG1]], %[[HOISTED_RELU2]])
  %res2 = mo.call @callsite_invariant(%arg1, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[CALL1_RESULT]], %[[CALL2_RESULT]])
  %res_add = mo.add(%res1, %res2) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %res_add : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Multiple boundary values from callsite-invariant arguments
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @multi_boundary_invariant
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG1:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG2:.+]]: !mo.tensor<[4], f32>)
mo.graph @multi_boundary_invariant(%arg0: !mo.tensor<[4], f32>, %arg1_invariant: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) isSubgraph {
  // Hoistable ops from callsite-invariant argument
  %0 = mo.relu(%arg1_invariant) : !mo.tensor<[4], f32>
  %1 = mo.relu(%arg1_invariant) : !mo.tensor<[4], f32>

  // Non-invariant ops using boundary values
  // CHECK: %[[ADD1_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG1]])
  %2 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>
  // CHECK: %[[ADD2_OUT:.+]] = mo.add(%[[ADD1_OUT]], %[[HOISTED_ARG2]])
  %3 = mo.add(%2, %1) : !mo.tensor<[4], f32>

  // CHECK: mo.output %[[ADD2_OUT]]
  mo.output %3 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_multi_boundary_invariant
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_multi_boundary_invariant(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst1 = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>

  // Hoisted ops for call 1
  // CHECK: %[[HOISTED_RELU1_1:.+]] = mo.relu(%[[CONST1]])
  // CHECK: %[[HOISTED_RELU1_2:.+]] = mo.relu(%[[CONST1]])
  // CHECK: %[[CALL1_RESULT:.+]] = mo.call @multi_boundary_invariant(%[[CALLER_ARG0]], %[[HOISTED_RELU1_1]], %[[HOISTED_RELU1_2]])
  %res1 = mo.call @multi_boundary_invariant(%arg0, %cst1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // Hoisted ops for call 2 (same invariant argument)
  // CHECK: %[[HOISTED_RELU2_1:.+]] = mo.relu(%[[CONST1]])
  // CHECK: %[[HOISTED_RELU2_2:.+]] = mo.relu(%[[CONST1]])
  // CHECK: %[[CALL2_RESULT:.+]] = mo.call @multi_boundary_invariant(%[[CALLER_ARG1]], %[[HOISTED_RELU2_1]], %[[HOISTED_RELU2_2]])
  %res2 = mo.call @multi_boundary_invariant(%arg1, %cst1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[CALL1_RESULT]], %[[CALL2_RESULT]])
  %res_add = mo.add(%res1, %res2) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %res_add : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Multiple callsites with different arguments - NO hoisting
// CHECK-LABEL: module
module {

// Graph should remain unchanged because arg1 is not callsite-invariant across callsites
// CHECK-LABEL: mo.graph @mixed_callsites
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @mixed_callsites(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) isSubgraph {
  // CHECK: %[[RELU_OUT:.+]] = mo.relu(%[[ARG1]])
  %0 = mo.relu(%arg1) : !mo.tensor<[4], f32>
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[RELU_OUT]])
  %1 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %1 : !mo.tensor<[4], f32>
}

// Caller should remain unchanged, no hoisting happens
// CHECK-LABEL: mo.graph @caller_mixed_callsites
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_mixed_callsites(%arg0: !mo.tensor<[4], f32>, %arg1_variable: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST:.+]] = mo.constant.external [name = "cst", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst = mo.constant.external [name = "cst", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>

  // Call 1: Uses constant
  // CHECK: %[[CALL1_RES:.+]] = mo.call @mixed_callsites(%[[CALLER_ARG0]], %[[CONST]])
  %res1 = mo.call @mixed_callsites(%arg0, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // Call 2: Uses variable (different value, so no callsite-invariant hoisting)
  // CHECK: %[[CALL2_RES:.+]] = mo.call @mixed_callsites(%[[CALLER_ARG0]], %[[CALLER_ARG1]])
  %res2 = mo.call @mixed_callsites(%arg0, %arg1_variable) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[ADD_RES:.+]] = mo.add(%[[CALL1_RES]], %[[CALL2_RES]])
  %res_add = mo.add(%res1, %res2) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_RES]]
  mo.output %res_add : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Complex invariant pattern with multiple invariant args
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @complex_invariant_pattern
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG:.+]]: !mo.tensor<[4], f32>)
mo.graph @complex_invariant_pattern(%arg0: !mo.tensor<[4], f32>, %arg1_invariant: !mo.tensor<[4], f32>, %arg2_invariant: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) isSubgraph {
  // These ops depend only on callsite-invariant args %arg1_invariant, %arg2_invariant
  %0 = mo.mul(%arg1_invariant, %arg2_invariant) : !mo.tensor<[4], f32>
  %1 = mo.relu(%0) : !mo.tensor<[4], f32>

  // This op depends on non-invariant arg0 and the hoisted boundary value
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG]])
  %2 = mo.add(%arg0, %1) : !mo.tensor<[4], f32>

  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %2 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_complex_invariant
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_complex_invariant(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst1 = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  // CHECK: %[[CONST2:.+]] = mo.constant.external [name = "cst2", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst2 = mo.constant.external [name = "cst2", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>

  // Hoisted ops for call 1
  // CHECK: %[[HOISTED_MUL1:.+]] = mo.mul(%[[CONST1]], %[[CONST2]])
  // CHECK: %[[HOISTED_RELU1:.+]] = mo.relu(%[[HOISTED_MUL1]])
  // CHECK: %[[CALL1_RESULT:.+]] = mo.call @complex_invariant_pattern(%[[CALLER_ARG0]], %[[HOISTED_RELU1]])
  %res1 = mo.call @complex_invariant_pattern(%arg0, %cst1, %cst2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // Hoisted ops for call 2 (same invariant arguments)
  // CHECK: %[[HOISTED_MUL2:.+]] = mo.mul(%[[CONST1]], %[[CONST2]])
  // CHECK: %[[HOISTED_RELU2:.+]] = mo.relu(%[[HOISTED_MUL2]])
  // CHECK: %[[CALL2_RESULT:.+]] = mo.call @complex_invariant_pattern(%[[CALLER_ARG1]], %[[HOISTED_RELU2]])
  %res2 = mo.call @complex_invariant_pattern(%arg1, %cst1, %cst2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[CALL1_RESULT]], %[[CALL2_RESULT]])
  %res_add = mo.add(%res1, %res2) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %res_add : !mo.tensor<[4], f32>
}

} // module

// -----

// Test that kgen.param.declare operations are hoisted correctly

// CHECK-LABEL: mo.graph @param_declare_op
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[seq_len], f32>, %[[HOISTED_ARG:.+]]: !mo.tensor<[seq_len], f32>)
mo.graph @param_declare_op<seq_len>(%arg0: !mo.tensor<[seq_len], f32>, %arg1_invariant: !mo.tensor<[seq_len], f32>, %arg2_invariant: !mo.tensor<[seq_len], f32>) -> (!mo.tensor<[seq_len], f32>) isSubgraph {

  // This assert can be hoisted as it only depends on seq_len
  kgen.param.declare ASSERT: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, identical(:scalar<si64> seq_len, 1))>

  %86 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %1715 = mo.slice<() -> NEW_DIM>(%arg0, %86, %86, %86) : (!mo.tensor<[seq_len], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[NEW_DIM], f32>

  // This assert cannot be hoisted as it depends on NEW_DIM, which is defined
  // by an operation that cannot be hoisted.
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, identical(:scalar<si64> NEW_DIM, 1))>
  kgen.param.declare ASSERT2: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, identical(:scalar<si64> NEW_DIM, 1))>

  // These ops depend only on callsite-invariant args %arg1_invariant, %arg2_invariant
  %0 = mo.mul(%arg1_invariant, %arg2_invariant) : !mo.tensor<[seq_len], f32>
  %1 = mo.relu(%0) : !mo.tensor<[seq_len], f32>

  // This op depends on non-invariant arg0 and the hoisted boundary value
  %2 = mo.add(%arg0, %1) : !mo.tensor<[seq_len], f32>

  mo.output %1 : !mo.tensor<[seq_len], f32>
}

// CHECK-LABEL: mo.graph @caller
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[seq_len], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[seq_len], f32>)
mo.graph @caller<seq_len>(%arg0: !mo.tensor<[seq_len], f32>, %arg1: !mo.tensor<[seq_len], f32>) -> (!mo.tensor<[seq_len], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst1 = mo.constant.external [name = "cst1", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[seq_len], f32>
  // CHECK: %[[CONST2:.+]] = mo.constant.external [name = "cst2", align = 1, device = #M.device_ref<"cpu", 0>]
  %cst2 = mo.constant.external [name = "cst2", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[seq_len], f32>

  // Hoisted ops for call 1
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, identical(:scalar<si64> seq_len, 1))>
  // CHECK: mo.call @param_declare_op
  %res1 = mo.call @param_declare_op(%arg0, %cst1, %cst2) : (!mo.tensor<[seq_len], f32>, !mo.tensor<[seq_len], f32>, !mo.tensor<[seq_len], f32>) -> !mo.tensor<[seq_len], f32>

  // Hoisted ops for call 2 (same invariant arguments)
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, identical(:scalar<si64> seq_len, 1))>
  // CHECK: mo.call @param_declare_op
  %res2 = mo.call @param_declare_op(%arg1, %cst1, %cst2) : (!mo.tensor<[seq_len], f32>, !mo.tensor<[seq_len], f32>, !mo.tensor<[seq_len], f32>) -> !mo.tensor<[seq_len], f32>

  %res_add = mo.add(%res1, %res2) : !mo.tensor<[seq_len], f32>
  mo.output %res_add : !mo.tensor<[seq_len], f32>
}

// -----

// Test case: Subgraph with multiple chains (trailing and non-trailing) - after
// hoisting, moveChainsToTheEnd ensures all chains end up trailing in the
// subgraph signature and at the call site. One mid chain is unused (not output)
// so the pass removes it; the other mid chain and trailing chain are output so
// preserved. Result: two trailing chains in signature and at call site.
// CHECK-LABEL: module
module {

// Subgraph has (invariant_tensor, ch_mid_unused, ch_mid, non_invariant_tensor,
// trailing_chain). ch_mid_unused is not used in body or output -> removed.
// ch_mid and ch_trailing are output -> preserved. After hoisting and
// moveChainsToTheEnd, signature is (non_invariant_tensor, boundary_tensor,
// ch_mid, trailing_chain) - two chains at end (unused mid chain removed).
// CHECK-LABEL: mo.graph @chains_non_trailing
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[ARG1:.+]]: !mo.tensor<[4], f32>, %[[CH0:.+]]: !mo.chain, %[[CH1:.+]]: !mo.chain)
mo.graph @chains_non_trailing(%arg0_invariant: !mo.tensor<[4], f32>, %ch_mid_unused: !mo.chain, %ch_mid: !mo.chain, %arg1: !mo.tensor<[4], f32>, %ch_trailing: !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain, !mo.chain) isSubgraph {
  %0 = mo.relu(%arg0_invariant) : !mo.tensor<[4], f32>
  %1 = mo.add(%0, %arg1) : !mo.tensor<[4], f32>
  mo.output %1, %ch_mid, %ch_trailing : !mo.tensor<[4], f32>, !mo.chain, !mo.chain
}

// Caller passes ch_unused for the unused mid chain; pass removes it so call
// has only 4 operands (tensors then two chains).
// CHECK-LABEL: mo.graph @caller_chains_non_trailing
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>, %[[CH_UNUSED:.+]]: !mo.chain, %[[CH1:.+]]: !mo.chain, %[[CH2:.+]]: !mo.chain)
mo.graph @caller_chains_non_trailing(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>, %ch_unused: !mo.chain, %ch1: !mo.chain, %ch2: !mo.chain) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.chain, !mo.chain) {
  // CHECK: %[[CST:.+]] = mo.constant.external
  %cst = mo.constant.external [name = "cst", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>

  // Unused mid chain (ch_unused) removed from call; operands are
  // (tensors, two chains) - 4 operands total.
  // CHECK: %[[HOISTED_RELU1:.+]] = mo.relu(%[[CST]])
  // CHECK: %[[CALL1:.+]] = mo.call @chains_non_trailing(%[[CALLER_ARG0]], %[[HOISTED_RELU1]], %[[CH1]], %[[CH2]])
  %res1, %out_ch_mid1, %out_ch_trailing1 = mo.call @chains_non_trailing(%cst, %ch_unused, %ch1, %arg0, %ch2) : (!mo.tensor<[4], f32>, !mo.chain, !mo.chain, !mo.tensor<[4], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain, !mo.chain)

  // CHECK: %[[HOISTED_RELU2:.+]] = mo.relu(%[[CST]])
  // CHECK: %[[CALL2:.+]] = mo.call @chains_non_trailing(%[[CALLER_ARG1]], %[[HOISTED_RELU2]], %[[CH1]], %[[CH2]])
  %res2, %out_ch_mid2, %out_ch_trailing2 = mo.call @chains_non_trailing(%cst, %ch_unused, %ch1, %arg1, %ch2) : (!mo.tensor<[4], f32>, !mo.chain, !mo.chain, !mo.tensor<[4], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain, !mo.chain)

  mo.output %res1, %res2, %out_ch_mid1, %out_ch_trailing1 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.chain, !mo.chain
}

} // module
