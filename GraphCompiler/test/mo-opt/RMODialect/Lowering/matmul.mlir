// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators --mo-shape-inference %s | FileCheck %s --enable-var-scope

mo.graph @matmul_static(%arg0: !mo.tensor<[1, 4], si64>, %arg1: !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], si64>, !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64>
    mo.output %1: !mo.tensor<[1, 1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_static([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[1, 4], si64>, !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 1], si64>

// -----

mo.graph @matmul_static_broadcast(%arg0: !mo.tensor<[10, 4, 1], si64>, %arg1: !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 4, 1], si64>, !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64>
    mo.output %1: !mo.tensor<[10, 4, 8], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_static_broadcast([[VARG0:[^ ]*]]: !mo.tensor<[10, 4, 1], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG1]]) : !mo.tensor<[1, 8], si64> -> !mo.tensor<[1, 1, 8], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, 1, 8], si64> -> !mo.tensor<[10, 1, 8], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.batch_matmul([[VARG0]], [[V1]]) : (!mo.tensor<[10, 4, 1], si64>, !mo.tensor<[10, 1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[10, 4, 8], si64>

// -----

mo.graph @matmul_complex<D1, D2>(%arg0: !mo.tensor<[10, D1, 7, D2], si64>, %arg1: !mo.tensor<[D1, D2, 100], si64>) -> !mo.tensor<[10, D1, 7, 100], si64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D1, 7, D2], si64>, !mo.tensor<[D1, D2, 100], si64>) -> !mo.tensor<[10, D1, 7, 100], si64>
    mo.output %1: !mo.tensor<[10, D1, 7, 100], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_complex<[[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[10, [[D1]], 7, [[D2]]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], 100], si64>) -> !mo.tensor<[10, [[D1]], 7, 100], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG1]]) : !mo.tensor<[[[D1]], [[D2]], 100], si64> -> !mo.tensor<[1, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, [[D1]], [[D2]], 100], si64> -> !mo.tensor<[10, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.batch_matmul([[VARG0]], [[V1]]) : (!mo.tensor<[10, [[D1]], 7, [[D2]]], si64>, !mo.tensor<[10, [[D1]], [[D2]], 100], si64>) -> !mo.tensor<[10, [[D1]], 7, 100], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[10, [[D1]], 7, 100], si64>

// -----

// The result dtype differs from the inputs, which are not cast.
mo.graph @matmul_wider_out_dtype(%arg0: !mo.tensor<[1, 4], bf16>, %arg1: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f32> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], bf16>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f32>
    mo.output %1: !mo.tensor<[1, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_wider_out_dtype([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], bf16>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[1, 4], bf16>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 1], f32>

// -----

mo.graph @matmul_narrower_out_dtype(%arg0: !mo.tensor<[1, 4], f32>, %arg1: !mo.tensor<[4, 1], f32>) -> !mo.tensor<[1, 1], bf16> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], f32>, !mo.tensor<[4, 1], f32>) -> !mo.tensor<[1, 1], bf16>
    mo.output %1: !mo.tensor<[1, 1], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_narrower_out_dtype([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], f32>) -> !mo.tensor<[1, 1], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[1, 4], f32>, !mo.tensor<[4, 1], f32>) -> !mo.tensor<[1, 1], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 1], bf16>

// -----

mo.graph @matmul_mixed_input_dtypes(%arg0: !mo.tensor<[1, 4], si8>, %arg1: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], si8>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16>
    mo.output %1: !mo.tensor<[1, 1], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_mixed_input_dtypes([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[1, 4], si8>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 1], bf16>

// -----

mo.graph @matmul_mixed_dtypes_with_reshape(%arg0: !mo.tensor<[7, 1, 4], f64>, %arg1: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[7, 1, 4], f64>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64>
    mo.output %1: !mo.tensor<[7, 1, 1], f64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_mixed_dtypes_with_reshape([[VARG0:[^ ]*]]: !mo.tensor<[7, 1, 4], f64>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG1]]) : !mo.tensor<[4, 1], bf16> -> !mo.tensor<[1, 4, 1], bf16>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, 4, 1], bf16> -> !mo.tensor<[7, 4, 1], bf16>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.batch_matmul([[VARG0]], [[V1]]) : (!mo.tensor<[7, 1, 4], f64>, !mo.tensor<[7, 4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[7, 1, 1], f64>
