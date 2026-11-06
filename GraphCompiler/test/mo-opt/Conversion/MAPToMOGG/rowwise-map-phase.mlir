// RUN: mo-opt %s --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s
// RUN: mo-opt %s --map-outline-structured-kernel-bodies --map-to-mogg | mo-opt

// A map phase lowers to a `mogg.rowwise.map` at the same place in the row,
// with its loads and stores at its own index.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

module {

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Mul registration = "mo.mul" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @scale
// CHECK:         mogg.call.rowwise shape(%{{.*}}) axis(1) context(%{{.*}}) (%[[ROW:[[:alnum:]_]+]] : !map.index<2>) capturing {%[[X:cap[0-9]*]] = %{{.*}} : !mogg.tensor, %[[OUT:cap[0-9]*]] = %{{.*}} : !mogg.tensor} device = "cpu" {
// CHECK-NEXT:      %[[ACC:.*]] = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:           %[[S:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Mul(%[[ACC]], %[[ACC]])
// CHECK-NEXT:      mogg.rowwise.map (%[[J:[[:alnum:]_]+]] : !map.index<2>) {
// CHECK-NEXT:        %[[V:.*]] = mogg.tensor.load %[[X]][%[[J]]]
// CHECK-NEXT:        %[[Y:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Mul(%[[V]], %[[S]])
// CHECK-NEXT:        mogg.tensor.store %[[OUT]][%[[J]]] = %[[Y]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
mo.graph @scale(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
    %s = map.call.elementwise @builtin_kernels.elementwise.Mul(%row, %acc, %acc) : !map.index<2>, f32, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %v = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %v, %s) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

}
