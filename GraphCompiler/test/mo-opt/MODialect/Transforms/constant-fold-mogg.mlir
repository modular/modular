// UNSUPPORTED: asan
// RUN: mo-opt --mo-constant-fold=iterations=1 %s | FileCheck %s

// COM: Examine dumped MO constant subgraph before execution
// RUN: mo-opt --mo-constant-fold=iterations=1 --mlir-print-ir-after=map-to-mogg 2>&1 %s | FileCheck %s -check-prefix=CHECK-CONST-MO-SUBGRAPH

// CHECK-CONST-MO-SUBGRAPH-LABEL: @partial_fold_bcast_constant_subgraphs
// CHECK-CONST-MO-SUBGRAPH: mo.static.broadcast_to

// the reshape is no longer decomposed; it stays a view call
// CHECK-CONST-MO-SUBGRAPH: mo.static.reshape

// COM: Do not constant fold broadcasts in the output, but allow them internally
// CHECK-LABEL: mo.graph
// CHECK-SAME: @partial_fold_bcast() -> !mo.tensor<[4, 2, 4], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : !M.array<8xf32>> : !mo.tensor<[1, 2, 4], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, 2, 4], f32> -> !mo.tensor<[4, 2, 4], f32>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[4, 2, 4], f32>
mo.graph @partial_fold_bcast() -> !mo.tensor<[4, 2, 4], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : !M.array<4xf32>> : !mo.tensor<[4], f32>

  // COM: Here we should constant fold up to the reshape.
  // COM: Note all of these increase memory but should be under the default
  // COM: threshold of 100 bytes.
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
  %2 = mo.static.reshape(%1): !mo.tensor<[2, 4], f32> -> !mo.tensor<[1, 2, 4], f32>

  // COM: Do not fold up to the below because it is a broadcast op.
  %3 = mo.static.broadcast_to(%2) : !mo.tensor<[1, 2, 4], f32> -> !mo.tensor<[4, 2, 4], f32>
  mo.output %3 : !mo.tensor<[4, 2, 4], f32>
}

// CHECK-CONST-MO-SUBGRAPH-LABEL: @nested_region_constant_subgraphs
// CHECK-CONST-MO-SUBGRAPH: mo.mul

// CHECK-LABEL: nested_region
mo.graph @nested_region(%g1 : !mo.tensor<[], bool>) -> (!mo.tensor<[2], si32>) {
  // CHECK-DAG: %[[G1:.*]] = mo.constant <#M.dense_array<6, 1>
  // CHECK-DAG: %[[G0:.*]] = mo.constant <#M.dense_array<0, 1>
  // CHECK-NEXT: mo.if
  %g2 = mo.if(%g1 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    // CHECK-NEXT: mo.yield %[[G0]]
    %g3 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
    %g4 = mo.mul(%g3, %g3) : !mo.tensor<[2], si32>
    mo.yield %g4 : !mo.tensor<[2], si32>
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: mo.yield %[[G1]]
    %g3 = mo.constant <#M.dense_array<6, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
    mo.yield %g3 : !mo.tensor<[2], si32>
  }
  mo.output %g2 : !mo.tensor<[2], si32>
}

// CHECK-CONST-MO-SUBGRAPH-LABEL: mo.graph @if_not_handled_constant_subgraphs()
// CHECK-CONST-MO-SUBGRAPH: mo.mul

// CHECK-LABEL: if_not_handled
mo.graph @if_not_handled(%e1 : !mo.tensor<[2], si32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2], si32>) {
  // CHECK-DAG: %[[E6:.*]] = mo.constant <#M.dense_array<0, 6>
  // CHECK-DAG: %[[E5:.*]] = mo.constant <#M.dense_array<0, 9>
  %e2 = mo.constant <#M.dense_array<0, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  // CHECK-NEXT: mo.if
  %e4 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %e5 = mo.mul(%e2, %e2) : !mo.tensor<[2], si32>
    // CHECK: mo.yield %[[E5]]
    mo.yield %e5 : !mo.tensor<[2], si32>
  } else {
    %e6 = mo.add(%e2, %e2) : !mo.tensor<[2], si32>
    // CHECK: mo.yield %[[E6]]
    mo.yield %e6 : !mo.tensor<[2], si32>
  }
  mo.output %e4 : !mo.tensor<[2], si32>
}

// CHECK-CONST-MO-SUBGRAPH-LABEL: mo.graph @multiple_subgraphs_constant_subgraphs
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.mul
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.sub

// CHECK-LABEL: mo.graph @multiple_subgraphs
// CHECK-SAME:(%[[R1:arg.*]]:
mo.graph @multiple_subgraphs(%r1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %r2 = mo.constant <#M.dense_array<1, -1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
  %r3 = mo.constant <#M.dense_array<6, 3> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>

  // CHECK-DAG: %[[R4:.*]] = mo.constant <#M.dense_array<1, 1>
  %r4 = mo.mul(%r2, %r2) : !mo.tensor<[2], si32>
  // CHECK-DAG: %[[R5:.*]] = mo.constant <#M.dense_array<5, 4>
  %r5 = mo.sub(%r3, %r2) : !mo.tensor<[2], si32>
  // CHECK-DAG: = mo.sub(%[[R1]], %[[R5]])
  %r6 = mo.sub(%r1, %r5) : !mo.tensor<[2], si32>
  // CHECK-DAG: = mo.sub(%[[R1]], %[[R4]])
  %r7 = mo.sub(%r1, %r4) : !mo.tensor<[2], si32>
  %r8 = mo.sub(%r6, %r7) : !mo.tensor<[2], si32>
  mo.output %r8 : !mo.tensor<[2], si32>
}

// CHECK-CONST-MO-SUBGRAPH-LABEL: mo.graph @all_const_constant_subgraphs
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.mul
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.sub
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.add

// CHECK-LABEL: @all_const
// CHECK-NEXT: %[[S1:.*]] = mo.constant <#M.dense_array<0, 5>
// CHECK-NEXT: mo.output %[[S1]]
mo.graph @all_const() -> (!mo.tensor<[2], si32>) {
  %s1 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
  %s2 = mo.constant <#M.dense_array<6, 3> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>

  %s3 = mo.mul(%s1, %s2) : !mo.tensor<[2], si32>
  %s4 = mo.sub(%s3, %s1) : !mo.tensor<[2], si32>
  %s5 = mo.add(%s4, %s3) : !mo.tensor<[2], si32>
  mo.output %s5 : !mo.tensor<[2], si32>
}

// No kernel for large_broadcast because it is not constant folded.
// CHECK-CONST-MO-SUBGRAPH-NOT: large_broadcast_constant_subgraphs()
mo.graph @large_broadcast() -> (!mo.tensor<[1024], si8>) {
  %shape = mo.constant <#M.dense_array<1024> : tensor<1xsi32>> :
			!mo.tensor<[1], si32>
  %data = mo.constant <#M.dense_array<1> : tensor<1xsi8>> :
			!mo.tensor<[1], si8>
  %broadcast = mo.broadcast_to(%data, %shape) :
     (!mo.tensor<[1], si8>, !mo.tensor<[1], si32>) -> !mo.tensor<[1024], si8>
  mo.output %broadcast : !mo.tensor<[1024], si8>
}
