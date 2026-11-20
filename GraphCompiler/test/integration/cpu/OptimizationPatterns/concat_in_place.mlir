// RUN: mo-opt %s --mo-shape-inference --mo-pattern-fusion | FileCheck %s --check-prefix=CHECK-IR
// RUN: mo-opt %s --gc-pipeline="constant-fold=false enable-post-abi-regions-fusion=true" | FileCheck %s --check-prefix=CHECK-MGP
// RUN: mo-opt %s --gc-pipeline="constant-fold=false enable-post-abi-regions-fusion=true" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-IR-LABEL: mo.graph @concat_static_output
// CHECK-IR-SAME:(%[[ARG0:.*]]:
// CHECK-IR: %[[VAL1:.*]] = mo.relu(%[[ARG0]])
// CHECK-IR: %[[VAL2:.*]] = mo.tanh(%[[ARG0]])
// CHECK-IR: %[[VAL3:.*]] = mogg.concat(%[[VAL1]], %[[VAL2]])
// CHECK-IR: mo.output %[[VAL3]]

// CHECK-MGP-LABEL: mef.model @concat_static_output

// CHECK-LABEL:--- Running 'concat_static_output':
// CHECK{LITERAL}: 'concat_static_output' returned tensor<6xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 7.615942e-01, 7.615942e-01, 7.615942e-01]

mo.graph @concat_static_output(%arg0: !mo.tensor<[3], f32>) -> (!mo.tensor<[6], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %0 = mo.relu(%arg0) : !mo.tensor<[3], f32>
  %1 = mo.tanh(%arg0) : !mo.tensor<[3], f32>
  %out = mo.concat[0](%0, %1) : (
    !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  ) -> !mo.tensor<[6], f32>
  mo.output %out : !mo.tensor<[6], f32>
}

// CHECK-IR-LABEL: mo.graph @concat_nested_output
// CHECK-IR:(%[[ARG0:.*]]:
// CHECK-IR: %[[VAL1:.*]] = mo.relu(%[[ARG0]])
// CHECK-IR: %[[VAL2:.*]] = mo.tanh(%[[ARG0]])
// CHECK-IR: %[[VAL3:.*]] = mogg.concat(%[[VAL1]], %[[VAL2]])
// CHECK-IR: %[[VAL4:.*]] = mo.sin(%[[ARG0]])
// CHECK-IR: %[[VAL5:.*]] = mo.concat[0] (%[[VAL3]], %[[VAL4]])
// CHECK-IR: mo.output %[[VAL5]]

// CHECK-MGP-LABEL: mef.model @concat_nested_output
// CHECK-LABEL:--- Running 'concat_nested_output':
// CHECK: 'concat_nested_output' returned tensor<9xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 7.6159{{.*}}e-01, 7.6159{{.*}}e-01, 7.6159{{.*}}e-01, 8.4147{{.*}}e-01, 8.4147{{.*}}e-01, 8.4147{{.*}}e-01]

mo.graph @concat_nested_output(%arg0: !mo.tensor<[3], f32>) -> (!mo.tensor<[9], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %0 = mo.relu(%arg0) : !mo.tensor<[3], f32>
  %1 = mo.tanh(%arg0) : !mo.tensor<[3], f32>
  %2 = mo.concat[0](%0, %1) : (
    !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  ) -> !mo.tensor<[6], f32>
  %3 = mo.sin(%arg0) : !mo.tensor<[3], f32>
  %out = mo.concat[0](%2, %3) : (
    !mo.tensor<[6], f32>, !mo.tensor<[3], f32>
  ) -> !mo.tensor<[9], f32>
  mo.output %out : !mo.tensor<[9], f32>
}

// CHECK-IR-LABEL: mo.graph @concat_within_if
// CHECK-IR-SAME: (%[[ARG0:.*]]: !mo.tensor<[3], f32>,
// CHECK-IR-SAME:  %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[6], f32>
// CHECK-IR:     %[[VAL4:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[6], f32> {
// CHECK-IR-DAG:   %[[VAL2:.*]] = mo.relu(%[[ARG0]]) : !mo.tensor<[3], f32>
// CHECK-IR-DAG:   %[[VAL3:.*]] = mo.tanh(%[[ARG0]]) : !mo.tensor<[3], f32>
// CHECK-IR-DAG:   %[[VAL5:.*]] = mogg.concat(%[[VAL2]], %[[VAL3]]) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[6], f32>
// CHECK-IR:       mo.yield %[[VAL5]] : !mo.tensor<[6], f32>
// CHECK-IR:     } else {
// CHECK-IR-DAG:   %[[VAL2:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[3], f32>
// CHECK-IR-DAG:   %[[VAL3:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[3], f32>
// CHECK-IR-DAG:   %[[VAL5:.*]] = mogg.concat(%[[VAL2]], %[[VAL3]]) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[6], f32>
// CHECK-IR:       mo.yield %[[VAL5]] : !mo.tensor<[6], f32>
// CHECK-IR:     }
// CHECK-IR:     mo.output %[[VAL4]] : !mo.tensor<[6], f32>

// CHECK-LABEL:--- Running 'concat_within_if':
// CHECK{LITERAL}: 'concat_within_if' returned tensor<6xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 7.615942e-01, 7.615942e-01, 7.615942e-01]

mo.graph @concat_within_if(%arg0: !mo.tensor<[3], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[6], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[6], f32> {
    %0 = mo.relu(%arg0) : !mo.tensor<[3], f32>
    %1 = mo.tanh(%arg0) : !mo.tensor<[3], f32>
    %2 = mo.concat[0](%0, %1) : (
      !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
    ) -> !mo.tensor<[6], f32>
    mo.yield %2 : !mo.tensor<[6], f32>
  } else {
    %0 = mo.exp(%arg0) : !mo.tensor<[3], f32>
    %1 = mo.erf(%arg0) : !mo.tensor<[3], f32>
    %2 = mo.concat[0](%0, %1) : (
      !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
    ) -> !mo.tensor<[6], f32>
    mo.yield %2 : !mo.tensor<[6], f32>
  }
  mo.output %2 : !mo.tensor<[6], f32>
}

// Don't optimize across regions

// CHECK-IR-LABEL:  mo.graph @concat_across_if
// CHECK-IR-SAME:   (%[[ARG0:.*]]: !mo.tensor<[3], f32>,
// CHECK-IR-SAME:    %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[6], f32>
// CHECK-IR-DAG:    %[[NEG:.*]] = mo.not(%[[COND]]) : !mo.tensor<[], bool>
// CHECK-IR-DAG:    %[[VAL1:.*]] = mo.relu(%[[ARG0]]) : !mo.tensor<[3], f32>
// CHECK-IR-DAG:    %[[VAL2:.*]] = mo.tanh(%[[ARG0]]) : !mo.tensor<[3], f32>
// CHECK-IR:        %[[VAL3:.*]] = mo.if(%[[NEG]] : !mo.tensor<[], bool>) -> !mo.tensor<[6], f32> {
// CHECK-IR:          %[[VAL4:.*]] = mo.concat[0] (%[[VAL1]], %[[VAL2]]) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[6], f32>
// CHECK-IR:          mo.yield %[[VAL4]] : !mo.tensor<[6], f32>
// CHECK-IR:        } else {
// CHECK-IR:          %[[VAL4:.*]] = mo.concat[0] (%[[VAL2]], %[[VAL1]]) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[6], f32>
// CHECK-IR:          mo.yield %[[VAL4]] : !mo.tensor<[6], f32>
// CHECK-IR:        }
// CHECK-IR:        mo.output %[[VAL3]] : !mo.tensor<[6], f32>

// Must be fully fused, i.e., has only one region in execute.
// CHECK-MGP-LABEL: mef.callable @concat_across_if$exec

// CHECK-LABEL:--- Running 'concat_across_if':
// CHECK{LITERAL}: 'concat_across_if' returned tensor<6xf32> [7.615942e-01, 7.615942e-01, 7.615942e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @concat_across_if(%arg0: !mo.tensor<[3], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[6], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %neg = mo.not(%cond) : !mo.tensor<[], bool>
  %0 = mo.relu(%arg0) : !mo.tensor<[3], f32>
  %1 = mo.tanh(%arg0) : !mo.tensor<[3], f32>
  %2 = mo.if(%neg : !mo.tensor<[], bool>) -> !mo.tensor<[6], f32> {
    %3 = mo.concat[0](%0, %1) : (
      !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
    ) -> !mo.tensor<[6], f32>
    mo.yield %3 : !mo.tensor<[6], f32>
  } else {
    %3 = mo.concat[0](%1, %0) : (
      !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
    ) -> !mo.tensor<[6], f32>
    mo.yield %3 : !mo.tensor<[6], f32>
  }
  mo.output %2 : !mo.tensor<[6], f32>
}

// CHECK-IR-LABEL: mo.graph @concat_multi_user_input
// CHECK-IR-SAME:(%[[ARG0:.*]]:
// CHECK-IR: %[[VAL1:.*]] = mo.relu(%[[ARG0]])
// CHECK-IR: %[[VAL2:.*]] = mo.tanh(%[[VAL1]])
// CHECK-IR: %[[VAL3:.*]] = mogg.concat(%[[VAL1]], %[[VAL2]])
// CHECK-IR: mo.output %[[VAL3]]

// CHECK-LABEL:--- Running 'concat_multi_user_input':
// CHECK{LITERAL}: 'concat_multi_user_input' returned tensor<6xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 7.615942e-01, 7.615942e-01, 7.615942e-01]

mo.graph @concat_multi_user_input(%arg0: !mo.tensor<[3], f32>) -> (!mo.tensor<[6], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %0 = mo.relu(%arg0) : !mo.tensor<[3], f32>
  %1 = mo.tanh(%0) : !mo.tensor<[3], f32>
  %out = mo.concat[0](%0, %1) : (
    !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  ) -> !mo.tensor<[6], f32>
  mo.output %out : !mo.tensor<[6], f32>
}

// CHECK-IR-LABEL: mo.graph @concat_duplicated_input
// CHECK-IR-SAME:(%[[ARG0:.*]]:
// CHECK-IR: %[[VAL1:.*]] = mo.tanh(%[[ARG0]])
// CHECK-IR: %[[VAL2:.*]] = mo.concat[0] (%[[VAL1]], %[[VAL1]])
// CHECK-IR: mo.output %[[VAL2]]

// CHECK-MGP-LABEL: mef.model @concat_duplicated_input

// CHECK-LABEL:--- Running 'concat_duplicated_input':
// CHECK{LITERAL}: 'concat_duplicated_input' returned tensor<6xf32> [7.615942e-01, 7.615942e-01, 7.615942e-01, 7.615942e-01, 7.615942e-01, 7.615942e-01]

mo.graph @concat_duplicated_input(%arg0: !mo.tensor<[3], f32>) -> (!mo.tensor<[6], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %0 = mo.tanh(%arg0) : !mo.tensor<[3], f32>
  %out = mo.concat[0](%0, %0) : (
    !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
  ) -> !mo.tensor<[6], f32>
  mo.output %out : !mo.tensor<[6], f32>
}
