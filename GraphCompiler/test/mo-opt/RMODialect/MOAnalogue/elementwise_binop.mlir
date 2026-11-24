// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @mod_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.mod([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @min_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.min([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @max_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.max([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @div_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.div([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @pow_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.pow([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @pow_mixed_precision<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_mixed_precision<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.pow([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @sub_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sub([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[D0, D1], f32>

// -----

mo.graph @mod_not_ok_bad_operands<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @mod_not_ok_bad_results<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %1 : !mo.tensor<[D0, D1], f64>
}

// -----

mo.graph @min_not_ok_bad_operands<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @min_not_ok_bad_results<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %1 : !mo.tensor<[D0, D1], f64>
}

// -----

mo.graph @max_not_ok_bad_operands<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @max_not_ok_bad_results<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %1 : !mo.tensor<[D0, D1], f64>
}

// -----

mo.graph @div_not_ok_bad_operands<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @div_not_ok_bad_results<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %1 : !mo.tensor<[D0, D1], f64>
}

// -----

mo.graph @sub_not_ok_bad_operands<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @sub_not_ok_bad_results<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f64>
  mo.output %1 : !mo.tensor<[D0, D1], f64>
}
