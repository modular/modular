// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @binary_comparison_ok<x>(%arg0: !mo.tensor<[x], f32>, %arg1: !mo.tensor<[x], f32>) -> !mo.tensor<[x], bool> {
  %1 = rmo.equal(%arg0, %arg1) : (!mo.tensor<[x], f32>, !mo.tensor<[x], f32>) -> !mo.tensor<[x], bool>
  mo.output %1 : !mo.tensor<[x], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_comparison_ok<x>([[VARG0:[^ ]*]]: !mo.tensor<[x], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[x], f32>) -> !mo.tensor<[x], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.equal([[VARG0]], [[VARG1]]) : (!mo.tensor<[x], f32>, !mo.tensor<[x], f32>) -> !mo.tensor<[x], bool>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[x], bool>

// -----

mo.graph @binary_comparison_ok_cast<x>(%arg0: !mo.tensor<[x], si8>, %arg1: !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bool> {
  %1 = rmo.not_equal(%arg0, %arg1) : (!mo.tensor<[x], si8>, !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bool>
  mo.output %1 : !mo.tensor<[x], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_comparison_ok_cast<x>([[VARG0:[^ ]*]]: !mo.tensor<[x], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.not_equal([[VARG0]], [[VARG1]]) : (!mo.tensor<[x], si8>, !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bool>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[x], bool>

// -----

// ui8 fits within a bf16 exactly. bf16 has 7 significand bits plus leading implicit 1 bit.
mo.graph @binary_comparison_ok_cast_ui8_bf16(%arg0: !mo.tensor<[7], ui8>, %arg1: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bool> {
    %1 = rmo.greater(%arg0, %arg1) : (!mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bool>
    mo.output %1: !mo.tensor<[7], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_comparison_ok_cast_ui8_bf16([[VARG0:[^ ]*]]: !mo.tensor<[7], ui8>, [[VARG1:[^ ]*]]: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.greater([[VARG0]], [[VARG1]]) : (!mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bool>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[7], bool>

// -----

mo.graph @binary_comparison_not_ok_bad_cast_ui8_si8(%arg0: !mo.tensor<[7], si8>, %arg1: !mo.tensor<[7], ui8>) -> !mo.tensor<[7], bool> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from ui8 to si8. Insert an explicit cast op if this conversion is wanted.}}
    // expected-error@below {{'rmo.equal' op failed to infer returned types}}
    %1 = rmo.equal(%arg0, %arg1) : (!mo.tensor<[7], si8>, !mo.tensor<[7], ui8>) -> !mo.tensor<[7], bool>
    mo.output %1: !mo.tensor<[7], bool>
}

// -----

mo.graph @binary_elemwise_param_static_bad_shape<x, y>(%arg0: !mo.tensor<[1, x], si64>, %arg1: !mo.tensor<[y, 5], si64>) -> !mo.tensor<[y, 5], bool> {
    // expected-error@below {{Input lhs (shape [1, x]) dimension at axis 1 (value x) and input rhs (shape [y, 5]) dimension at axis 1 (value 5) are neither equivalent nor broadcastable.}}
    // expected-error@below {{'rmo.greater' op failed to infer returned types}}
    %1 = rmo.greater(%arg0, %arg1) : (!mo.tensor<[1, x], si64>, !mo.tensor<[y, 5], si64>) -> !mo.tensor<[y, 5], bool>
    mo.output %1: !mo.tensor<[y, 5], bool>
}
