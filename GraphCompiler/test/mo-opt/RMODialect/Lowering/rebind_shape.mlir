// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators %s | FileCheck %s --enable-var-scope

mo.graph @rebind_tensor_shape_mix_mix<D1 -> D2>(%arg0: !mo.tensor<[4, D1], f32>) -> !mo.tensor<[D2, 3], f32> {
    %0 = mo.rebind<() -> D3>(%arg0) : !mo.tensor<[4, D1], f32> -> !mo.tensor<[4, D3], f32>
    kgen.param.declare RMO_SH_EQ5: scalar<bool> = <and(identical(:scalar<si64> D3, 3), identical(:scalar<si64> D4, 4))>
    kgen.param.declare RMO_SH_CH6: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed with error: \22RMO rebind shape check failed.\22" : !kgen.string}, RMO_SH_EQ5)>
    %1 = mo.rebind<() -> D4>(%0) : !mo.tensor<[4, D3], f32> -> !mo.tensor<[D4, 3], f32>
    mo.output<D4> %1 : !mo.tensor<[D4, 3], f32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_mix_mix<[[D1:[^ ]*]] -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[4, [[D1]]], f32>) -> !mo.tensor<[[[D2]], 3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[4, [[D1]]], f32> -> !mo.tensor<[4, [[D3]]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: scalar<bool> = <and(identical(:scalar<si64> [[D3]], 3), identical(:scalar<si64> [[D4:[^ ]*]], 4))>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "{{.*}}" : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D4]]>([[V0]]) : !mo.tensor<[4, [[D3]]], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V1]] : !mo.tensor<[[[D4]], 3], f32>
