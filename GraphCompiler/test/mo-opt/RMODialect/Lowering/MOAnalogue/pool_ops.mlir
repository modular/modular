// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @max_pool_dyn_dims<D0, D1, D2, D3, D4, D5 -> D6, D7, D8, D9>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %fs: !mo.tensor<[D4], si64>, %st: !mo.tensor<[D4], si64>, %di: !mo.tensor<[D4], si64>, %pa: !mo.tensor<[D5], si64>) -> !mo.tensor<[D6, D7, D8, D9], f32> {
  %0 = rmo.mo.max_pool<() -> D10, D11, D12, D13>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4], si64>, !mo.tensor<[D4], si64>, !mo.tensor<[D4], si64>, !mo.tensor<[D5], si64>) -> !mo.tensor<[D10, D11, D12, D13], f32>
  mo.output<D10, D11, D12, D13> %0 : !mo.tensor<[D10, D11, D12, D13], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_dyn_dims
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[ARG1]], strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]]]
//  CHECK-NEXT:   mo.output{{.*}} [[V0]]

// -----

mo.graph @max_pool_static<() -> D0, D1, D2, D3>(%arg0: !mo.tensor<[3, 8, 43, 49], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], f32> {
  %0 = rmo.mo.max_pool<() -> D4, D5, D6, D7>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, 8, 43, 49], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[D4, D5, D6, D7], f32>
  mo.output<D4, D5, D6, D7> %0 : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[3, 8, 43, 49], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[4], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[ARG1]], strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]]]
//  CHECK-NEXT:   mo.output{{.*}} [[V0]]

// -----

mo.graph @max_pool_static_dyn<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[3, D0, 43, D1], f32>, %fs: !mo.tensor<[D2], si64>, %st: !mo.tensor<[D2], si64>, %di: !mo.tensor<[D2], si64>, %pa: !mo.tensor<[D3], si64>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %0 = rmo.mo.max_pool<() -> D8, D9, D10, D11>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, D0, 43, D1], f32>, !mo.tensor<[D2], si64>, !mo.tensor<[D2], si64>, !mo.tensor<[D2], si64>, !mo.tensor<[D3], si64>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %0 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static_dyn
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[3, {{[A-Za-z0-9_]+}}, 43, {{[A-Za-z0-9_]+}}], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[ARG1]], strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]]]
//  CHECK-NEXT:   mo.output{{.*}} [[V0]]
