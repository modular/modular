// UNSUPPORTED: ubsan

// RUN: mo-opt %s --mo-add-fallback-shape-funcs | FileCheck %s

// resize.bicubic gained a fallback shape-function registration (its op
// already implements MO_ShapeMaterialization). The fallback must invoke
// the "mo.resize.bicubic" shape function with the size operand marked
// data-dependent, mirroring resize.linear/nearest.
mo.graph @resize_bicubic_fallback<() -> D0, D1>(
  %input: !mo.tensor<[1, 3, 224, 224], f32>,
  %size: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1], f32> {

  %res = mo.resize.bicubic<() -> D2, D3>(%input, %size) : (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph @resize_bicubic_fallback
// CHECK: mo.invoke_shape_func["mo.resize.bicubic"]
// CHECK-SAME: [dataDeptTensors = [1 : i32]]
