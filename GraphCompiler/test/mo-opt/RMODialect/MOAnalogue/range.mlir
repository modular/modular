// RUN: mo-opt --split-input-file --verify-diagnostics

mo.graph @range<() -> D0>(%arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32> {
  %0 = rmo.mo.range<() -> D1>(%arg0, %arg1, %arg2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D1], si32>
  mo.output<D1> %0 : !mo.tensor<[D1], si32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @range<() -> [[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[], si32>, [[VARG1:[^ ]*]]: !mo.tensor<[], si32>, [[VARG2:[^ ]*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D1]]], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.range<() -> [[D2:[^ ]*]]>([[VARG0]], [[VARG1]], [[VARG2]]) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D2]]], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind([[V0]]) : !mo.tensor<[[[D2]]], si32> -> !mo.tensor<[[[D1]]], si32>
// CHECK-NEXT: mo.output<[[D2]]> [[V0]] : !mo.tensor<[[[D2]]], si32>

// -----

mo.graph @mismatched_types(%arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32> {
  // expected-error@below {{'rmo.range' op failed to verify that all of {start, limit, step, result} have same dtype}}
  %0 = rmo.mo.range(%arg0, %arg1, %arg2) : (!mo.tensor<[], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32>
  mo.output %0 : !mo.tensor<[D0], si32>
}
