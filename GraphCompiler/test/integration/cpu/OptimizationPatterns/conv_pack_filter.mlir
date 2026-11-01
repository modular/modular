// RUN: mo-opt %s --mo-shape-inference --gc-pipeline="constant-fold=false" -o %t_new_api.mlir
// RUN: mt --execute --input-shapes=4x3x3x1 --input-shapes=2 -model-inputs=ones --result-output-style=full %t_new_api.mlir | FileCheck %s

// Note that the MGP checks are to make sure the `pack_matmul_b_shape_func` call
// got folded away at compile time, so the kernel can use static shape from the
// packed weights.

// CHECK-EXEC-LABEL: --- Running 'conv_const_filter':
// CHECK-IR-LABEL: conv_const_filter
// CHECK-MGP-LABEL: conv_const_filter
mo.graph @conv_const_filter(
  %input: !mo.tensor<[4, 3, 3, 1], f32>,
  %dummy: !mo.tensor<[2], si64>
) -> !mo.tensor<[4, 2, 2, 1], f32> {

  // CHECK-IR: mo.invoke_shape_func["pack_conv_filter_shape"]<() -> [[P1:.*]], [[P2:.*]], [[P3:.*]], [[P4:.*]], [[P5:.*]]>(%{{.*}}) [kgenParams = {dilations = dense<1> : vector<2xi64>, filter_shape = dense<[2, 2, 1, 1]> : vector<4xi64>, input_shape = dense<[2, 2, 1, 1]> : vector<4xi64>, num_groups = 1 : index, output_shape = dense<[4, 2, 2, 1]> : vector<4xi64>, paddings = dense<0> : vector<4xi64>, strides = dense<1> : vector<2xi64>}]
  // CHECK-IR: %[[PACKED_FILTER:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {num_groups = 1 : index}] : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[P4]], [[P5]]], f32, {layout = #mo.layout<FRSCf>}>
  // CHECK-IR: mo.conv({{.*}}, %[[PACKED_FILTER]])
  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %filter = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK-MGP-NOT: "pack_conv_filter_shape"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv:{{.*}}"
  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 1], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 1], f32>
}

// CHECK{LITERAL}: 'conv_const_filter' returned tensor<4x2x2x1xf32> [1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01]


// CHECK-EXEC-LABEL: --- Running 'conv_transp_filter':
// CHECK-IR-LABEL: conv_transp_filter
// CHECK-MGP-LABEL: conv_transp_filter
mo.graph @conv_transp_filter(
  %input: !mo.tensor<[4, 3, 3, 1], f32>,
  %dummy: !mo.tensor<[2], si64>
) -> !mo.tensor<[4, 2, 2, 1], f32> {

  // CHECK-IR: mo.invoke_shape_func["pack_conv_filter_shape"]<() -> [[P1:.*]], [[P2:.*]], [[P3:.*]], [[P4:.*]], [[P5:.*]]>(%{{.*}}) [kgenParams = {dilations = dense<1> : vector<2xi64>, filter_shape = dense<[2, 2, 1, 1]> : vector<4xi64>, input_shape = dense<[2, 2, 1, 1]> : vector<4xi64>, num_groups = 1 : index, output_shape = dense<[4, 2, 2, 1]> : vector<4xi64>, paddings = dense<0> : vector<4xi64>, strides = dense<1> : vector<2xi64>}]
  // CHECK-IR: %[[PACKED_FILTER:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {num_groups = 1 : index}] : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[P4]], [[P5]]], f32, {layout = #mo.layout<FRSCf>}>
  // CHECK-IR: mo.conv({{.*}}, %[[PACKED_FILTER]])
  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %filter = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>
  %perm = mo.constant <#M.dense_array<1, 0, 2, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %filter_t = mo.transpose(%filter, %perm) : (!mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[4], si64>) -> !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>

  // CHECK-MGP-NOT: "pack_conv_filter_shape"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv:{{.*}}"
  %res = mo.conv(%input, %filter_t) [strides = %ones, dilations = %ones, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 1], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 1], f32>
}

// CHECK{LITERAL}: 'conv_transp_filter' returned tensor<4x2x2x1xf32> [1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01]

// CHECK-EXEC-LABEL: --- Running 'conv_const_filter_dyn_output':
// CHECK-IR-LABEL: conv_const_filter_dyn_output
// CHECK-MGP-LABEL: conv_const_filter_dyn_output
mo.graph @conv_const_filter_dyn_output<() -> D0, D1>(
  %input: !mo.tensor<[4, 3, 3, 1], f32>,
  %strides: !mo.tensor<[2], si64>
) -> !mo.tensor<[4, D0, D1, 1], f32> {

  // CHECK-IR: mo.invoke_shape_func["pack_conv_filter_shape"]<() -> [[P1:.*]], [[P2:.*]], [[P3:.*]], [[P4:.*]], [[P5:.*]]>(%{{.*}}) [kgenParams = {dilations = dense<1> : vector<2xi64>, filter_shape = dense<[2, 2, 1, 1]> : vector<4xi64>, input_shape = dense<[2, 2, 1, 1]> : vector<4xi64>, num_groups = 1 : index, output_shape = dense<[4, -9223372036854775808, -9223372036854775808, 1]> : vector<4xi64>, paddings = dense<0> : vector<4xi64>, strides = dense<-9223372036854775808> : vector<2xi64>}]
  // CHECK-IR: %[[PACKED_FILTER:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {num_groups = 1 : index}] : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[P4]], [[P5]]], f32, {layout = #mo.layout<FRSCf>}>
  // CHECK-IR: mo.conv({{.*}}, %[[PACKED_FILTER]])
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %filter = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK-MGP-NOT: "pack_conv_filter_shape"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv:{{.*}}"
  %res = mo.conv<() -> D2, D3>(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, D2, D3, 1], f32>

  mo.output<D2, D3> %res : !mo.tensor<[4, D2, D3, 1], f32>
}

// CHECK{LITERAL}: 'conv_const_filter_dyn_output' returned tensor<4x2x2x1xf32> [1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01]
