// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @abs_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.abs(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @abs_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.abs([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @cos_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.cos(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cos_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.cos([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @erf_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.erf(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @erf_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.erf([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @exp_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.exp(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @exp_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.exp([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @floor_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.floor(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @floor_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.floor([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @log_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.log(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.log([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @log1p_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.log1p(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log1p_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.log1p([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @logsoftmax_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.reduce.logsoftmax(%arg0) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @logsoftmax_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.reduce.logsoftmax([[VARG0]]) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @negative_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.negative(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @negative_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.negative([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @not_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.not(%arg0) : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @not_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.not([[VARG0]]) : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D0, D1], bool>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], bool>

// -----

mo.graph @relu_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.relu(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @relu_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.relu([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @round_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.round(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @round_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.round([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @rsqrt_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.rsqrt(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rsqrt_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.rsqrt([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @sin_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.sin(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sin_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sin([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @sqrt_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.sqrt(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sqrt_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sqrt([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @tanh_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.tanh(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @tanh_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.tanh([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @trunc_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.trunc(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @trunc_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.trunc([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @not_non_bool_dtypes<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1{{'rmo.mo.not' op operand #0 must be boolean element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.not(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @trunc_mismatched_dtype<D0, D1>(%arg0: !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1{{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.trunc(%arg0) : (!mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
