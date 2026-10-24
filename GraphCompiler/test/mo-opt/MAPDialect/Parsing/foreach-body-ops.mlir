// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

// map.call.elementwise verifies its callee resolves to a mogg.kernel.decl
// with an `elementwise` function, so this stub must be present.
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @foreachBodyStatic
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>, [[Y:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[8, 16, 128], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[8, 16, 128]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<3>)
// CHECK-NEXT:        capturing {[[X0:%[^ ,]+]] = [[X]] : !mo.tensor<[8, 16, 128], f32>, [[Y0:%[^ ,]+]] = [[Y]] : !mo.tensor<[8, 16, 128], f32>, [[OUT0:%[^ ,]+]] = [[OUT]] : !mo.tensor<[8, 16, 128], f32>} {
// CHECK-NEXT:      [[XV:%[^ ]+]] = map.load [[X0]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:      [[YV:%[^ ]+]] = map.load [[Y0]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:      [[SUM:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX]], [[XV]], [[YV]]) : !map.index<3>, f32, f32 -> f32
// CHECK-NEXT:      map.store [[OUT0]][[[IDX]]] = [[SUM]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  mo.output [[RESULT]] : !mo.tensor<[8, 16, 128], f32>
mo.graph @foreachBodyStatic(%x: !mo.tensor<[8, 16, 128], f32>,
                            %y: !mo.tensor<[8, 16, 128], f32>)
    -> !mo.tensor<[8, 16, 128], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[8, 16, 128], f32>, %y : !mo.tensor<[8, 16, 128], f32>)
      outputs (%out : !mo.tensor<[8, 16, 128], f32>) {
    map.iter.elementwise.body<[8, 16, 128]>
        index (%idx: !map.index<3>)
        capturing {%x0 = %x : !mo.tensor<[8, 16, 128], f32>, %y0 = %y : !mo.tensor<[8, 16, 128], f32>, %out0 = %out : !mo.tensor<[8, 16, 128], f32>} {
      %xv = map.load %x0[%idx] : !mo.tensor<[8, 16, 128], f32>
      %yv = map.load %y0[%idx] : !mo.tensor<[8, 16, 128], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<3>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[8, 16, 128], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[8, 16, 128], f32>
}

// CHECK-LABEL: mo.graph @foreachBodyDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>, [[Y2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[D0, D1], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[D0, D1]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}} = [[X2]] : !mo.tensor<[D0, D1], f32>, {{.*}} = [[Y2]] : !mo.tensor<[D0, D1], f32>, {{.*}} = [[OUT2]] : !mo.tensor<[D0, D1], f32>} {
mo.graph @foreachBodyDynamic<D0, D1>(
    %x: !mo.tensor<[D0, D1], f32>,
    %y: !mo.tensor<[D0, D1], f32>)
    -> !mo.tensor<[D0, D1], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[D0, D1], f32>, %y : !mo.tensor<[D0, D1], f32>)
      outputs (%out : !mo.tensor<[D0, D1], f32>) {
    map.iter.elementwise.body<[D0, D1]>
        index (%idx: !map.index<2>)
        capturing {%x0 = %x : !mo.tensor<[D0, D1], f32>, %y0 = %y : !mo.tensor<[D0, D1], f32>, %out0 = %out : !mo.tensor<[D0, D1], f32>} {
      %xv = map.load %x0[%idx] : !mo.tensor<[D0, D1], f32>
      %yv = map.load %y0[%idx] : !mo.tensor<[D0, D1], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[D0, D1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[D0, D1], f32>
}

}
