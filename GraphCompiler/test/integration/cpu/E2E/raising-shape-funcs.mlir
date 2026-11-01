// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: not mt --execute --model-inputs=ones --on-failure=continue --result-output-style=full %t.mlir | FileCheck %s

// ===----------------------------------------------------------------------===//
// Shape Function Error Cases
// ===----------------------------------------------------------------------===//
// This test file validates that shape functions properly raise errors during
// graph execution via emit-mojo. Note: mt stops execution after the first error,
// so only the first test will actually execute in a single test run.

// CHECK-LABEL: --- Running 'range_zero_step_error':
// CHECK: 'range_zero_step_error' returned <error:
// CHECK: {{.*}}step must be non-zero
mo.graph @range_zero_step_error(%start_in: !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32> {
  // start_in from input (will be 1), subtract 1 to get 0
  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %start = mo.sub(%start_in, %one) : !mo.tensor<[], si32>
  %stop = mo.constant <#M.dense_array<10> : tensor<si32>> : !mo.tensor<[], si32>
  %step = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>

  %result = mo.range<() -> D0>(%start, %stop, %step) :
    (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32>

  mo.output %result : !mo.tensor<[D0], si32>
}

// The following tests are valid but won't execute because mt stops after the first error:

mo.graph @range_invalid_bounds_pos_step_error(%step_in: !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32> {
  // step comes from input (will be 1), start > stop triggers error
  %start = mo.constant <#M.dense_array<10> : tensor<si32>> : !mo.tensor<[], si32>
  %stop = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>

  %result = mo.range<() -> D0>(%start, %stop, %step_in) :
    (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32>

  mo.output %result : !mo.tensor<[D0], si32>
}

mo.graph @range_invalid_bounds_neg_step_error(%stop_in: !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32> {
  // stop_in from input (will be 1), multiply by 10 to get 10
  %ten = mo.constant <#M.dense_array<10> : tensor<si32>> : !mo.tensor<[], si32>
  %stop = mo.mul(%stop_in, %ten) : !mo.tensor<[], si32>
  %start = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %step = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>

  %result = mo.range<() -> D0>(%start, %stop, %step) :
    (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32>

  mo.output %result : !mo.tensor<[D0], si32>
}

mo.graph @slice_zero_step_error(%start: !mo.tensor<[1], si32>) -> !mo.tensor<[D0], f32> {
  // start comes from graph input (will be [1]), step=0 triggers error
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %stop = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %step = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %result = mo.slice<() -> D0>(%input, %start, %stop, %step) :
    (!mo.tensor<[5], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>)
    -> !mo.tensor<[D0], f32>

  mo.output %result : !mo.tensor<[D0], f32>
}
