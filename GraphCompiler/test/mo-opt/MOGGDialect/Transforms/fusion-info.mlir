// RUN: mo-opt --gc-pipeline %s --mlir-print-ir-before=jit --mlir-print-local-scope -o %t.mlir 2>&1 | FileCheck %s

// Check that fusionInfo is printed on kernel stubs after fusion passes.
// Under the new API we expect the matmul+add to be fused followed by sequence fusing the cumsum.
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_sequence<[{{.*}}#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul", loc = loc({{.*}}:{{[0-9]+}}:{{[0-9]+}}){{.*}}#mogg.fusion_leaf<"mo.add", loc = loc({{.*}}:{{[0-9]+}}:{{[0-9]+}}){{.*}}operandIndices = [0]>
mo.graph @fusion_info_simple(
    %arg0: !mo.tensor<[20, 30], f32>,
    %arg1: !mo.tensor<[30, 40], f32>,
    %arg2: !mo.tensor<[20, 40], f32>
) -> !mo.tensor<[20, 40], f32>{
    %matmul_res = mo.matmul(
        %arg0,
        %arg1
    ): (
        !mo.tensor<[20, 30], f32>,
        !mo.tensor<[30, 40], f32>
    ) -> !mo.tensor<[20, 40], f32>

    %bias_res = mo.add(%matmul_res, %arg2): !mo.tensor<[20, 40], f32>

    %final_res = mo.cumsum(%bias_res) [axis = 1, exclusive = 1, reverse = 0]: (
        !mo.tensor<[20, 40], f32>
    ) -> !mo.tensor<[20, 40], f32>
    mo.output %final_res: !mo.tensor<[20, 40], f32>
}
