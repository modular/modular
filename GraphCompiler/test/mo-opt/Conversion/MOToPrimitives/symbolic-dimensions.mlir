// RUN: mo-opt %s --mogg-to-mgp | FileCheck %s

// There should be two assertions in the primary graph checking that D0 is equal
// to the other occurrences of D0 and no checks in the subgraph version.

// CHECK-LABEL: mgp.model @dims_must_match_same_tensor
// CHECK: %{{.*}} = mgp.assert %{{.*}}, "symbolic dimension 'D0' for input 0 does not match prior uses of that dimension"
// CHECK: %{{.*}} = mgp.assert %{{.*}}, "symbolic dimension 'D0' for input 0 does not match prior uses of that dimension"
mo.graph @dims_must_match_same_tensor<D0>(%arg0: !mo.tensor<[D0, D0, D0], f32>) -> !mo.tensor<[D0, D0, D0], f32> {
  mo.output %arg0 : !mo.tensor<[D0, D0, D0], f32>
}

// CHECK-LABEL: mgp.model @dims_must_match_same_tensor_subgraph
// CHECK-NOT: mgp.assert
mo.graph @dims_must_match_same_tensor_subgraph<D0>(%arg0: !mo.tensor<[D0, D0, D0], f32>) -> !mo.tensor<[D0, D0, D0], f32> isSubgraph {
  mo.output %arg0 : !mo.tensor<[D0, D0, D0], f32>
}
