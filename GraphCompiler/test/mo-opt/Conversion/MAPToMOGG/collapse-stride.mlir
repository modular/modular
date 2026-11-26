// RUN: mo-opt %s --mogg-import-packages --map-to-mogg | FileCheck %s

// A collapse view lowers to `mogg.tensor.create.collapse` over the tensor it
// views, keeping its groups and its static out-shape. It can be stored into,
// so over the kernel's output it is mutable.

// CHECK-LABEL: mo.graph @collapseStatic
// CHECK:       [[IT:%[^ ]+]] = mogg.tensor.create({{.*}}io = Input
// CHECK:       [[OT:%[^ ]+]] = mogg.tensor.create({{.*}}io = Output
// CHECK:       mogg.tensor.create.collapse([[IT]]) <elementType = f32, mut = false, outShape = #mosh<ape[6, 4]> : !mosh.ape, groups = [2, 1]> : !mogg.tensor
// CHECK:       mogg.tensor.create.collapse([[OT]]) <elementType = f32, mut = true, outShape = #mosh<ape[6, 4]> : !mosh.ape, groups = [2, 1]> : !mogg.tensor
mo.graph @collapseStatic(%arg0: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[2, 3, 4], f32>)
    outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %1 = map.index.collapse.stride<#map.view.signature<[2, 3, 4] -> [6, 4]>, groups = [2, 1]> (%arg0) : !mo.tensor<[2, 3, 4], f32> -> !mo.tensor<[6, 4], f32>
    %2 = map.index.collapse.stride<#map.view.signature<[2, 3, 4] -> [6, 4]>, groups = [2, 1]> (%out) : !mo.tensor<[2, 3, 4], f32> -> !mo.tensor<[6, 4], f32>
    map.iter.elementwise.body<[6, 4]>
      index   (%arg2: !map.index<2>)
      capturing {%arg3 = %1 : !mo.tensor<[6, 4], f32>, %arg4 = %2 : !mo.tensor<[6, 4], f32>} {
      %3 = map.load %arg3[%arg2] : !mo.tensor<[6, 4], f32>
      map.store %arg4[%arg2] = %3 : !mo.tensor<[6, 4], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.copy", loc = loc(unknown), sig = (!mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32>>>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[2, 3, 4], f32>
}

// A symbolic merged extent stays the graph parameter it was declared as; the
// run-time extent comes from the viewed tensor's own shape.
// CHECK-LABEL: mo.graph @collapseDynamic
// CHECK:       mogg.tensor.create.collapse({{.*}}) <elementType = f32, mut = false, outShape = #mosh<ape[param_flat_1]> : !mosh.ape, groups = [3]> : !mogg.tensor
// CHECK:       mogg.tensor.create.collapse({{.*}}) <elementType = f32, mut = true, outShape = #mosh<ape[param_flat_1]> : !mosh.ape, groups = [3]> : !mogg.tensor
mo.graph @collapseDynamic<B>(%arg0: !mo.tensor<[B, 3, 4], f32>) -> !mo.tensor<[B, 3, 4], f32> {
  kgen.param.declare param_flat_1: scalar<si64> = <mul_no_wrap(B, 12)>
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[B, 3, 4], f32>)
    outputs (%out : !mo.tensor<[B, 3, 4], f32>) {
    %1 = map.index.collapse.stride<#map.view.signature<[B, 3, 4] -> [param_flat_1]>, groups = [3]> (%arg0) : !mo.tensor<[B, 3, 4], f32> -> !mo.tensor<[param_flat_1], f32>
    %2 = map.index.collapse.stride<#map.view.signature<[B, 3, 4] -> [param_flat_1]>, groups = [3]> (%out) : !mo.tensor<[B, 3, 4], f32> -> !mo.tensor<[param_flat_1], f32>
    map.iter.elementwise.body<[param_flat_1]>
      index   (%arg2: !map.index<1>)
      capturing {%arg3 = %1 : !mo.tensor<[param_flat_1], f32>, %arg4 = %2 : !mo.tensor<[param_flat_1], f32>} {
      %3 = map.load %arg3[%arg2] : !mo.tensor<[param_flat_1], f32>
      map.store %arg4[%arg2] = %3 : !mo.tensor<[param_flat_1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.copy", loc = loc(unknown), sig = (!mo.tensor<[B, 3, 4], f32>) -> !mo.tensor<[B, 3, 4], f32>>>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[B, 3, 4], f32>
}
