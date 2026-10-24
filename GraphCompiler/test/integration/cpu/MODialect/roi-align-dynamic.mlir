// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute --input-shapes=1x5x5x1 --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @roi_align<D0, D1, D2, D3 -> D4, D5, D6, D7>(%inp_ones: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
    %output_height = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %rois = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 2.0, 2.0> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>

    %output = mo.roi_align<() -> D8, D9, D10, D11>(%inp_ones, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
    mo.output<D8, D9, D10, D11> %output : !mo.tensor<[D8, D9, D10, D11], f32>
}
// Ones, not zeros: a zero-filled input passed whether or not the kernel wrote
// the output at all.
// CHECK{LITERAL}: 'roi_align' returned tensor<1x2x2x1xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
