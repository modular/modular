// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @conv_dyn_params<D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11 -> D12, D13, D14, D15, D16>(%input : !mo.tensor<[D0, D1, D2, D3, D4], f32>, %filter : !mo.tensor<[D5, D6, D7, D8, D9], f32>, %st: !mo.tensor<[D10], si64>, %di: !mo.tensor<[D10], si64>, %pa: !mo.tensor<[D11], si64>, %ng: !mo.tensor<[], si64>) -> (!mo.tensor<[D12, D13, D14, D15, D16], f32>) {
  %0 = rmo.mo.conv<() -> D17, D18, D19, D20, D21>(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[D0, D1, D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7, D8, D9], f32>, !mo.tensor<[D10], si64>, !mo.tensor<[D10], si64>, !mo.tensor<[D11], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D17, D18, D19, D20, D21], f32>
  mo.output<D17, D18, D19, D20, D21> %0 : !mo.tensor<[D17, D18, D19, D20, D21], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_dyn_params<D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11 -> D12, D13, D14, D15, D16>([[ARG0:%.*]]: !mo.tensor<[D0, D1, D2, D3, D4], f32>, [[ARG1:%.*]]: !mo.tensor<[D5, D6, D7, D8, D9], f32>, [[ARG2:%.*]]: !mo.tensor<[D10], si64>, [[ARG3:%.*]]: !mo.tensor<[D10], si64>, [[ARG4:%.*]]: !mo.tensor<[D11], si64>, [[ARG5:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[D12, D13, D14, D15, D16], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.conv<() -> D17, D18, D19, D20, D21>([[ARG0]], [[ARG1]]) [strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]], num_groups = [[ARG5]], filter_layout = ""] : (!mo.tensor<[D0, D1, D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7, D8, D9], f32>, !mo.tensor<[D10], si64>, !mo.tensor<[D10], si64>, !mo.tensor<[D11], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D17, D18, D19, D20, D21], f32>
//  CHECK-NEXT:   mo.output<D17, D18, D19, D20, D21> [[V0]] : !mo.tensor<[D17, D18, D19, D20, D21], f32>
