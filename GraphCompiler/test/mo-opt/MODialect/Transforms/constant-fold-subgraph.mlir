// RUN: mo-opt %s --mo-constant-fold-subgraph-test | FileCheck %s

// CHECK-LABEL: all_const
// CHECK-NEXT: %[[S1:.*]] = mo.constant <#M.dense_array<0, 1>
// CHECK-NEXT: %[[S2:.*]] = mo.constant <#M.dense_array<6, 3>
// CHECK-NEXT: %[[S3:.*]] = mo.mul
// CHECK-NEXT: %[[S4:.*]] = mo.sub
// CHECK-NEXT: %[[S5:.*]] = mo.add
// CHECK: subgraph %[[S1]] %[[S2]] %[[S3]] %[[S4]] %[[S5]]
// CHECK: outputs %[[S5]]
// CHECK-NOT: module
mo.graph @all_const() -> (!mo.tensor<[2], si32>) {
  %s1 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
  %s2 = mo.constant <#M.dense_array<6, 3> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>

  %s3 = mo.mul(%s1, %s2) : !mo.tensor<[2], si32>
  %s4 = mo.sub(%s3, %s1) : !mo.tensor<[2], si32>
  %s5 = mo.add(%s3, %s4) : !mo.tensor<[2], si32>
  mo.output %s5 : !mo.tensor<[2], si32>
}

// CHECK: multiple_subgraphs(%[[R1:.*]]:
// CHECK-NEXT: %[[R2:.*]] = mo.constant <#M.dense_array<0, 1>
// CHECK-NEXT: %[[R3:.*]] = mo.constant <#M.dense_array<6, 3>
// CHECK-NEXT: %[[R4:.*]] = mo.mul
// CHECK-NEXT: %[[R5:.*]] = mo.sub
// CHECK-NEXT: %[[R6:.*]] = mo.sub
// CHECK-NEXT: %[[R7:.*]] = mo.sub
// CHECK-NEXT: %[[R8:.*]] = mo.sub
// CHECK-DAG: subgraph %[[R2]] %[[R4]]
// CHECK-DAG: outputs %[[R4]]
// CHECK-DAG: subgraph %[[R3]] %[[R5]]
// CHECK-DAG: outputs %[[R5]]
// CHECK-NOT: module
mo.graph @multiple_subgraphs(%r1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %r2 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
  %r3 = mo.constant <#M.dense_array<6, 3> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>

  %r4 = mo.mul(%r2, %r2) : !mo.tensor<[2], si32>
  %r5 = mo.sub(%r3, %r3) : !mo.tensor<[2], si32>
  %r6 = mo.sub(%r1, %r5) : !mo.tensor<[2], si32>
  %r7 = mo.sub(%r1, %r4) : !mo.tensor<[2], si32>
  %r8 = mo.sub(%r6, %r7) : !mo.tensor<[2], si32>
  mo.output %r8 : !mo.tensor<[2], si32>
}

// CHECK: if_not_handled(%[[E1:.*]]:
// CHECK-NEXT: %[[E2:.*]] = mo.constant <#M.dense_array<0, 1>
// CHECK-NEXT: %[[E3:.*]] = mo.constant <#M.dense_array<1>
// CHECK-NEXT: %[[E4:.*]] = mo.if
// CHECK-NEXT: %[[E5:.*]] = mo.mul
// CHECK-DAG: %[[E6:.*]] = mo.add
// CHECK: subgraph %[[E2]] %[[E5]] %[[E6]]
// CHECK: outputs %[[E5]] %[[E6]]
mo.graph @if_not_handled(%e1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %e2 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
  %e3 = mo.constant <#M.dense_array<1> : tensor<1xi8>> :
				!mo.tensor<[], bool>

  %e4 = mo.if(%e3 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %e5 = mo.mul(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e5 : !mo.tensor<[2], si32>
  } else {
    %e6 = mo.add(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e6 : !mo.tensor<[2], si32>
  }
  mo.output %e4 : !mo.tensor<[2], si32>
}

// CHECK-LABEL: nested_region
// CHECK-DAG: %[[G3:.*]] = mo.constant <#M.dense_array<0, 1>
// CHECK-DAG: %[[G4:.*]] = mo.mul(%[[G3]], %[[G3]])
// CHECK: subgraph %[[G3]] %[[G4]]
// CHECK: outputs %[[G4]]
mo.graph @nested_region(%g1 : !mo.tensor<[], bool>) -> (!mo.tensor<[2], si32>) {
  %g2 = mo.if(%g1 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %g3 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
    %g4 = mo.mul(%g3, %g3) : !mo.tensor<[2], si32>
    mo.yield %g4 : !mo.tensor<[2], si32>
  } else {
    %g3 = mo.constant <#M.dense_array<6, 1> : tensor<2xsi32>> :
				!mo.tensor<[2], si32>
    mo.yield %g3 : !mo.tensor<[2], si32>
  }
  mo.output %g2 : !mo.tensor<[2], si32>
}

// Output parameters should not be constant folded.
// CHECK-LABEL: parameter_outputs
// CHECK-NOT: subgraph
mo.graph @parameter_outputs<M>() -> (!mo.tensor<[M], si8>) {
  %data = mo.constant <#M.dense_array<1024> : tensor<1xsi8>> :
			!mo.tensor<[1], si8>
  %broadcast = mo.static.broadcast_to(%data) :!mo.tensor<[1], si8> -> !mo.tensor<[M], si8>
  mo.output %broadcast : !mo.tensor<[M], si8>
}

// CHECK-LABEL: repeated_application
// CHECK: %[[E1:.*]] = mo.constant <#M.dense_array<3>
// CHECK: %[[E2:.*]] = mo.constant <#M.dense_array<2>
// CHECK: %[[E3:.*]] = mo.add(%[[E1]], %[[E1]])
// CHECK: %[[E4:.*]] = mo.greater_equal(%[[E3]], %[[E2]])
// CHECK: %[[E5:.*]] = mo.select(%[[E4]], %[[E3]]
// CHECK: %[[E6:.*]] = mo.sub(%[[E5]], %[[E2]])
// CHECK: subgraph %[[E1]] %[[E2]] %[[E3]] %[[E4]]
// CHECK: outputs %[[E3]] %[[E4]]
mo.graph @repeated_application(%e0 : !mo.tensor<[], si32>) -> !mo.tensor<[], si32> {
  %e1 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> :
				!mo.tensor<[], si32>
  %e2 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> :
				!mo.tensor<[], si32>
  %e3 = mo.add(%e1, %e1) : !mo.tensor<[], si32>
  %e4 = mo.greater_equal(%e3, %e2) :(!mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], bool>
  %e5 = mo.select(%e4, %e3, %e0) :
    (!mo.tensor<[], bool>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], si32>
  %e6 = mo.sub(%e5, %e2) : !mo.tensor<[], si32>
  mo.output %e6 : !mo.tensor<[], si32>
}

// CHECK-LABEL: shape_func
// CHECK: %[[G1:.*]] = mo.constant <#M.dense_array<1.
// CHECK: %[[G2:.*]] = mo.static.broadcast_to(%[[G1]])
// CHECK: subgraph %[[G1]] %[[G2]]
// CHECK: outputs %[[G2]]
mo.graph @shape_func() -> (!mo.tensor<[?, ?], f32>) {
  %g1 = mo.constant <#M.dense_array<1.> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %g2 = mo.static.broadcast_to(%g1):!mo.tensor<[1], f32> -> !mo.tensor<[768,768], f32, {layout = #mo.layout<KN>}>
  mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> MMPackSh0, MMPackSh1>(%g2) [kgenParams = {transpose_in_0 = false}] : (!mo.tensor<[768, 768], f32, {layout = #mo.layout<KN>}>)
  %g3 = mo.layout.transform(%g2) : !mo.tensor<[768, 768], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[MMPackSh0, MMPackSh1], f32, {layout = #mo.layout<KNkni>}>
  %g4 = mo.rebind(%g3): !mo.tensor<[MMPackSh0, MMPackSh1], f32, {layout = #mo.layout<KNkni>}> -> !mo.tensor<[?, ?], f32>
  mo.output %g4: !mo.tensor<[?, ?], f32>
}

// make sure that we are not matching anything in the mlir of the result
// CHECK: module
