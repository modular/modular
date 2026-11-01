// RUN: mo-opt --mlir-print-local-scope %s --map-outline-structured-kernel-bodies | FileCheck %s

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

// map.call.elementwise verifies its callee resolves to a mogg.kernel.decl
// with an `elementwise` function, so this stub must be present.
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @add
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X:%[^ :,]+]]: !mo.tensor<[256, 256], f32>, [[Y:%[^ :,]+]]: !mo.tensor<[256, 256], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[256, 256], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[256, 256]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {[[X0:%[^ ,]+]] = [[X]] : !mo.tensor<[256, 256], f32>, [[Y0:%[^ ,]+]] = [[Y]] : !mo.tensor<[256, 256], f32>, [[OUT0:%[^ ,]+]] = [[OUT]] : !mo.tensor<[256, 256], f32>} {
// CHECK-NEXT:      [[XV:%[^ ]+]] = map.load [[X0]][[[IDX]]] : !mo.tensor<[256, 256], f32>
// CHECK-NEXT:      [[YV:%[^ ]+]] = map.load [[Y0]][[[IDX]]] : !mo.tensor<[256, 256], f32>
// CHECK-NEXT:      [[RES:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX]], [[XV]], [[YV]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:      map.store [[OUT0]][[[IDX]]] = [[RES]] : !mo.tensor<[256, 256], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>, hasReadEffect = false, hasWriteEffect = false}
// CHECK-NEXT:  mo.output [[RESULT]] : !mo.tensor<[256, 256], f32>
mo.graph @add(%arg0: !mo.tensor<[256, 256], f32>, %arg1: !mo.tensor<[256, 256], f32>) -> !mo.tensor<[256, 256], f32> {
  %0 = map.iter.elementwise<[256, 256]>
      inputs  (%arg0: !mo.tensor<[256, 256], f32>, %arg1: !mo.tensor<[256, 256], f32>)
      outputs (%out: !mo.tensor<[256, 256], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[256, 256], f32>
  {
    %xv = map.load %arg0[%idx] : !mo.tensor<[256, 256], f32>
    %yv = map.load %arg1[%idx] : !mo.tensor<[256, 256], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[256, 256], f32>
  } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[256, 256], f32>
}

// CHECK-LABEL: mo.graph @addDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X2:%[^ :,]+]]: !mo.tensor<[M, N], f32>, [[Y2:%[^ :,]+]]: !mo.tensor<[M, N], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[M, N], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[M, N]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {[[X02:%[^ ,]+]] = [[X2]] : !mo.tensor<[M, N], f32>, [[Y02:%[^ ,]+]] = [[Y2]] : !mo.tensor<[M, N], f32>, [[OUT02:%[^ ,]+]] = [[OUT2]] : !mo.tensor<[M, N], f32>} {
// CHECK-NEXT:      [[XV2:%[^ ]+]] = map.load [[X02]][[[IDX2]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:      [[YV2:%[^ ]+]] = map.load [[Y02]][[[IDX2]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:      [[RES2:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX2]], [[XV2]], [[YV2]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:      map.store [[OUT02]][[[IDX2]]] = [[RES2]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>, hasReadEffect = false, hasWriteEffect = false}
// CHECK-NEXT:  mo.output [[RESULT2]] : !mo.tensor<[M, N], f32>
mo.graph @addDynamic<M, N>(%arg0: !mo.tensor<[M, N], f32>, %arg1: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = map.iter.elementwise<[M, N]>
      inputs  (%arg0: !mo.tensor<[M, N], f32>, %arg1: !mo.tensor<[M, N], f32>)
      outputs (%out: !mo.tensor<[M, N], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[M, N], f32>
  {
    %xv = map.load %arg0[%idx] : !mo.tensor<[M, N], f32>
    %yv = map.load %arg1[%idx] : !mo.tensor<[M, N], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[M, N], f32>
  } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[M, N], f32>
}

// map.iter.opaque is left untouched.
// CHECK-LABEL: mo.graph @alreadyOpaque
// CHECK:         [[RESULT3:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X3:%[^ :,]+]]: !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    outputs ([[OUT3:%[^ :,]+]] : !mo.tensor<[2, 2], f32>) {
// CHECK-NEXT:  }
// CHECK-NEXT:  mo.output [[RESULT3]] : !mo.tensor<[2, 2], f32>
mo.graph @alreadyOpaque(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = map.iter.opaque inputs (%arg0 : !mo.tensor<[2, 2], f32>) outputs (%out : !mo.tensor<[2, 2], f32>) {
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

}
