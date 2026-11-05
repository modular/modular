// RUN: mo-opt %s --pass-pipeline='builtin.module(map-outline-structured-kernel-bodies, mo.graph(map-livm, map.iter.opaque(canonicalize,cse)))' | FileCheck %s

// A broadcast load in a rowwise kernel's map phase hoists into a strided view
// captured by the outlined body, like one in an elementwise kernel; the phase
// then reads it at its own index.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

module {

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Mul registration = "mo.mul" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @scaleByGamma
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X:%[^ :,]+]]: !mo.tensor<[8, 128], f32>, [[G:%[^ :,]+]]: !mo.tensor<[128], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[8, 128], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.broadcast.stride<#map.index.broadcast.spec<[128] -> [8, 128]>> ([[G]])
// CHECK-NEXT:      map.iter.rowwise.body<[8, 128], axis = 1>
// CHECK-NEXT:        row ([[ROW:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing ([[X0:%[^ ,]+]] = [[X]] : !mo.tensor<[8, 128], f32>, {{.*}}[[OUT0:%[^ ,]+]] = [[OUT]] : !mo.tensor<[8, 128], f32>, [[GV:%[^ ,]+]] = [[STRIDED]] : !mo.tensor<[8, 128], f32>) {
// CHECK:           map.iter.map index ([[I:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:         map.index.broadcast
// CHECK:             map.load [[GV]][[[I]]] : !mo.tensor<[8, 128], f32>
// CHECK:             map.store [[OUT0]][[[I]]]
mo.graph @scaleByGamma(%x: !mo.tensor<[8, 128], f32>, %gamma: !mo.tensor<[128], f32>)
    -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %gamma: !mo.tensor<[128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>)
      -> !mo.tensor<[8, 128], f32>
  {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %xv : f32
    }
    map.iter.map index (%i: !map.index<2>) {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %gi = map.index.broadcast<#map.index.broadcast.spec<[128] -> [8, 128]>> (%i)
                : !map.index<2> -> !map.index<1>
      %gv = map.load %gamma[%gi] : !mo.tensor<[128], f32>
      %n = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %acc)
               : !map.index<2>, f32, f32 -> f32
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %n, %gv)
               : !map.index<2>, f32, f32 -> f32
      map.store %out[%i] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

}
