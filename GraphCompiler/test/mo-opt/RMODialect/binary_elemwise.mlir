// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @binary_elemwise_ok<x>(%arg0: !mo.tensor<[x], f32>, %arg1: !mo.tensor<[x], f32>) -> !mo.tensor<[x], f32> {
  %1 = rmo.add(%arg0, %arg1) : (!mo.tensor<[x], f32>, !mo.tensor<[x], f32>) -> !mo.tensor<[x], f32>
  mo.output %1 : !mo.tensor<[x], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_ok<x>([[VARG0:[^ ]*]]: !mo.tensor<[x], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[x], f32>) -> !mo.tensor<[x], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.add([[VARG0]], [[VARG1]]) : (!mo.tensor<[x], f32>, !mo.tensor<[x], f32>) -> !mo.tensor<[x], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[x], f32>

// -----

mo.graph @binary_elemwise_ok_cast<x>(%arg0: !mo.tensor<[x], si8>, %arg1: !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bf16> {
  %1 = rmo.sub(%arg0, %arg1) : (!mo.tensor<[x], si8>, !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bf16>
  mo.output %1 : !mo.tensor<[x], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_ok_cast<x>([[VARG0:[^ ]*]]: !mo.tensor<[x], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.sub([[VARG0]], [[VARG1]]) : (!mo.tensor<[x], si8>, !mo.tensor<[x], bf16>) -> !mo.tensor<[x], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[x], bf16>

// -----

// ui8 fits within a bf16 exactly. bf16 has 7 significand bits plus leading implicit 1 bit.
mo.graph @binary_elemwise_ok_cast_ui8_bf16(%arg0: !mo.tensor<[7], ui8>, %arg1: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bf16> {
    %1 = rmo.mul(%arg0, %arg1) : (!mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bf16>
    mo.output %1: !mo.tensor<[7], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_ok_cast_ui8_bf16([[VARG0:[^ ]*]]: !mo.tensor<[7], ui8>, [[VARG1:[^ ]*]]: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mul([[VARG0]], [[VARG1]]) : (!mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[7], bf16>

// -----

mo.graph @binary_elemwise_not_ok_bad_cast_ui8_si8(%arg0: !mo.tensor<[7], si8>, %arg1: !mo.tensor<[7], ui8>) -> !mo.tensor<[7], si8> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from ui8 to si8. Insert an explicit cast op if this conversion is wanted.}}
    // expected-error@below {{'rmo.mod' op failed to infer returned types}}
    %1 = rmo.mod(%arg0, %arg1) : (!mo.tensor<[7], si8>, !mo.tensor<[7], ui8>) -> !mo.tensor<[7], si8>
    mo.output %1: !mo.tensor<[7], si8>
}

// -----

mo.graph @binary_elemwise_not_ok_bad_cast_f16_bf16(%arg0: !mo.tensor<[7], bf16>, %arg1: !mo.tensor<[7], f16>) -> !mo.tensor<[7], bf16> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from f16 to bf16. Insert an explicit cast op if this conversion is wanted.}}
    // expected-error@below {{'rmo.min' op failed to infer returned types}}
    %1 = rmo.min(%arg0, %arg1) : (!mo.tensor<[7], bf16>, !mo.tensor<[7], f16>) -> !mo.tensor<[7], bf16>
    mo.output %1: !mo.tensor<[7], bf16>
}

// -----

mo.graph @binary_elemwise_dyn_static_bad_shape<x, y>(%arg0: !mo.tensor<[1, x], si64>, %arg1: !mo.tensor<[y, 5], si64>) -> !mo.tensor<[y, 5], si64> {
    // expected-error@below {{Input lhs (shape [1, x]) dimension at axis 1 (value x) and input rhs (shape [y, 5]) dimension at axis 1 (value 5) are neither equivalent nor broadcastable.}}
    // expected-error@below {{'rmo.mul' op failed to infer returned types}}
    %1 = rmo.mul(%arg0, %arg1) : (!mo.tensor<[1, x], si64>, !mo.tensor<[y, 5], si64>) -> !mo.tensor<[y, 5], si64>
    mo.output %1: !mo.tensor<[y, 5], si64>
}

// -----

mo.graph @binary_elemwise_dyn_dyn_bad_shape<x, y>(%arg0: !mo.tensor<[x], si64>, %arg1: !mo.tensor<[4, y], si64>) -> !mo.tensor<[4, x], si64> {
    // expected-error@below {{Input lhs (shape [x]) dimension at axis 0 (value x) and input rhs (shape [4, y]) dimension at axis 1 (value y) are neither equivalent nor broadcastable.}}
    // expected-error@below {{'rmo.mul' op failed to infer returned types}}
    %1 = rmo.mul(%arg0, %arg1) : (!mo.tensor<[x], si64>, !mo.tensor<[4, y], si64>) -> !mo.tensor<[4, x], si64>
    mo.output %1: !mo.tensor<[4, x], si64>
}
