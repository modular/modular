// RUN: mo-opt %s | FileCheck %s

// Fusion info prints as an alias named after the fused ops, and the tree is
// spelled out once at the top. Past 32 characters the name keeps whole ops and
// counts the ones it dropped.

// CHECK-DAG: #matmul_gelu_mul = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul">, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.gelu">, #mogg.fusion_leaf<"mo.mul">]>], operandIndices = [0]>>
// CHECK-DAG: #matmul_gelu_mul_add_sub_relu_and2 = #mogg.fusion<#mogg.fusion_sequence<
// CHECK-DAG: #shapeof_range = #mogg.fusion<#mogg.fusion_leaf<"shapeof(mo.range)">>

// CHECK-LABEL: mogg.func @short private
// CHECK-SAME: fusionInfo = #matmul_gelu_mul}
mogg.func @short private () attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul">, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.gelu">, #mogg.fusion_leaf<"mo.mul">]>], operandIndices = [0]>>} {
  mogg.output
}

// CHECK-LABEL: mogg.func @long private
// CHECK-SAME: fusionInfo = #matmul_gelu_mul_add_sub_relu_and2}
mogg.func @long private () attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_sequence<[#mogg.fusion_leaf<"mo.matmul">, #mogg.fusion_leaf<"mo.gelu">, #mogg.fusion_leaf<"mo.mul">, #mogg.fusion_leaf<"mo.add">, #mogg.fusion_leaf<"mo.sub">, #mogg.fusion_leaf<"mo.relu">, #mogg.fusion_leaf<"mo.cast">, #mogg.fusion_leaf<"mo.exp">]>>} {
  mogg.output
}

// CHECK-LABEL: mogg.func @nested private
// CHECK-SAME: fusionInfo = #shapeof_range}
mogg.func @nested private () attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"shapeof(mo.range)">>} {
  mogg.output
}
