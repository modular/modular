// RUN: mo-opt %s --canonicalize | FileCheck %s

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

module {

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Mul registration = "mo.mul" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// A rowwise binding the same value twice is deduped to one input, and both
// loads in the fold read the surviving arg.
// CHECK-LABEL: mo.graph @dedupesRepeatedRowwiseInput
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  ([[A:%[^ :]+]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:  outputs
// CHECK:         map.load [[A]][
// CHECK:         map.load [[A]][
mo.graph @dedupesRepeatedRowwiseInput(%x: !mo.tensor<[8, 128], f32>)
    -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x as %a: !mo.tensor<[8, 128], f32>,
               %x as %b: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %av = map.load %a[%i] : !mo.tensor<[8, 128], f32>
      %bv = map.load %b[%i] : !mo.tensor<[8, 128], f32>
      %ab = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %av, %bv)
                : !map.index<2>, f32, f32 -> f32
      map.yield %ab : f32
    }
    map.store %out[%row] = %acc : !mo.tensor<[8, 1], f32>
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

}
