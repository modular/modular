// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators --mo-shape-inference %s | FileCheck %s --enable-var-scope

mo.graph @binary_elemwise_static(%arg0: !mo.tensor<[1, 4], si64>, %arg1: !mo.tensor<[1, 4], si64>) -> !mo.tensor<[1, 4], si64> {
    %1 = rmo.add(%arg0, %arg1) : (!mo.tensor<[1, 4], si64>, !mo.tensor<[1, 4], si64>) -> !mo.tensor<[1, 4], si64>
    mo.output %1: !mo.tensor<[1, 4], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_static([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 4], si64>) -> !mo.tensor<[1, 4], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.add([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 4], si64>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 4], si64>

// -----

mo.graph @binary_elemwise_static_broadcast(%arg0: !mo.tensor<[10, 4, 1], si64>, %arg1: !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64> {
    %1 = rmo.sub(%arg0, %arg1) : (!mo.tensor<[10, 4, 1], si64>, !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64>
    mo.output %1: !mo.tensor<[10, 4, 8], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_static_broadcast([[VARG0:[^ ]*]]: !mo.tensor<[10, 4, 1], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[10, 4, 1], si64> -> !mo.tensor<[10, 4, 8], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[1, 8], si64> -> !mo.tensor<[10, 4, 8], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.sub([[V0]], [[V1]]) : !mo.tensor<[10, 4, 8], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[10, 4, 8], si64>

// -----

mo.graph @binary_elemwise_complex<D1, D2>(%arg0: !mo.tensor<[10, D1, D2, 1], si64>, %arg1: !mo.tensor<[D1, D2, 100], si64>) -> !mo.tensor<[10, D1, D2, 100], si64> {
    %1 = rmo.min(%arg0, %arg1) : (!mo.tensor<[10, D1, D2, 1], si64>, !mo.tensor<[D1, D2, 100], si64>) -> !mo.tensor<[10, D1, D2, 100], si64>
    mo.output %1: !mo.tensor<[10, D1, D2, 100], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_complex<[[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[10, [[D1]], [[D2]], 1], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], 100], si64>) -> !mo.tensor<[10, [[D1]], [[D2]], 100], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[10, [[D1]], [[D2]], 1], si64> -> !mo.tensor<[10, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D1]], [[D2]], 100], si64> -> !mo.tensor<[10, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.min([[V0]], [[V1]]) : !mo.tensor<[10, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[10, [[D1]], [[D2]], 100], si64>

// -----

mo.graph @binary_elemwise_dyn_static_1_broadcast<D1, D2>(%arg0: !mo.tensor<[D1, 1], si64>, %arg1: !mo.tensor<[1, D2], si64>) -> !mo.tensor<[D1, D2], si64> {
    %1 = rmo.max(%arg0, %arg1) : (!mo.tensor<[D1, 1], si64>, !mo.tensor<[1, D2], si64>) -> !mo.tensor<[D1, D2], si64>
    mo.output %1: !mo.tensor<[D1, D2], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_dyn_static_1_broadcast<[[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], 1], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[1, [[D2]]], si64>) -> !mo.tensor<[[[D1]], [[D2]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D1]], 1], si64> -> !mo.tensor<[[[D1]], [[D2]]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[1, [[D2]]], si64> -> !mo.tensor<[[[D1]], [[D2]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.max([[V0]], [[V1]]) : !mo.tensor<[[[D1]], [[D2]]], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[[[D1]], [[D2]]], si64>

// -----

mo.graph @binary_elemwise_real_world_condition(%arg0: !mo.tensor<[1, 10, 1, 10], si64>, %arg1: !mo.tensor<[50, 20, 1, 20, 1], si64>) -> !mo.tensor<[50, 20, 10, 20, 10], si64> {
    %1 = rmo.pow(%arg0, %arg1) : (!mo.tensor<[1, 10, 1, 10], si64>, !mo.tensor<[50, 20, 1, 20, 1], si64>) -> !mo.tensor<[50, 20, 10, 20, 10], si64>
    mo.output %1: !mo.tensor<[50, 20, 10, 20, 10], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_real_world_condition([[VARG0:[^ ]*]]: !mo.tensor<[1, 10, 1, 10], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[50, 20, 1, 20, 1], si64>) -> !mo.tensor<[50, 20, 10, 20, 10], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[1, 10, 1, 10], si64> -> !mo.tensor<[50, 20, 10, 20, 10], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[50, 20, 1, 20, 1], si64> -> !mo.tensor<[50, 20, 10, 20, 10], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.pow([[V0]] : !mo.tensor<[50, 20, 10, 20, 10], si64>, [[V1]] : !mo.tensor<[50, 20, 10, 20, 10], si64>) : !mo.tensor<[50, 20, 10, 20, 10], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[50, 20, 10, 20, 10], si64>

// -----

mo.graph @binary_elemwise_dtype_cast_int_float(%arg0: !mo.tensor<[1, 4], si8>, %arg1: !mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], bf16> {
    %1 = rmo.add(%arg0, %arg1) : (!mo.tensor<[1, 4], si8>, !mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], bf16>
    mo.output %1: !mo.tensor<[1, 4], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_dtype_cast_int_float([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG0]]) : (!mo.tensor<[1, 4], si8>) -> !mo.tensor<[1, 4], bf16>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.add([[V0]], [[VARG1]]) : !mo.tensor<[1, 4], bf16>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 4], bf16>


// -----

mo.graph @binary_elemwise_dtype_cast_unsigned_signed(%arg0: !mo.tensor<[1, 4], si16>, %arg1: !mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], si16> {
    %1 = rmo.sub(%arg0, %arg1) : (!mo.tensor<[1, 4], si16>, !mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], si16>
    mo.output %1: !mo.tensor<[1, 4], si16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_dtype_cast_unsigned_signed([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si16>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], si16> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG1]]) : (!mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], si16>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.sub([[VARG0]], [[V0]]) : !mo.tensor<[1, 4], si16>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 4], si16>

// -----

mo.graph @binary_elemwise_dtype_cast_bool_other(%arg0: !mo.tensor<[1, 4], bool>, %arg1: !mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], ui8> {
    %1 = rmo.mul(%arg0, %arg1) : (!mo.tensor<[1, 4], bool>, !mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], ui8>
    mo.output %1: !mo.tensor<[1, 4], ui8>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_dtype_cast_bool_other([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], bool>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 4], ui8>) -> !mo.tensor<[1, 4], ui8> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG0]]) : (!mo.tensor<[1, 4], bool>) -> !mo.tensor<[1, 4], ui8>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.mul([[V0]], [[VARG1]]) : !mo.tensor<[1, 4], ui8>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 4], ui8>

// -----

mo.graph @binary_elemwise_dtype_cast_float_float(%arg0: !mo.tensor<[1, 4], f64>, %arg1: !mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], f64> {
    %1 = rmo.div(%arg0, %arg1) : (!mo.tensor<[1, 4], f64>, !mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], f64>
    mo.output %1: !mo.tensor<[1, 4], f64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_dtype_cast_float_float([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], f64>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], f64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG1]]) : (!mo.tensor<[1, 4], bf16>) -> !mo.tensor<[1, 4], f64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.div([[VARG0]], [[V0]]) : !mo.tensor<[1, 4], f64>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 4], f64>
