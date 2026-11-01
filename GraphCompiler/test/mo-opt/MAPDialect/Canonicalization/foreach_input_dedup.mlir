// RUN: mo-opt %s --canonicalize | FileCheck %s

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// A foreach binding the same value twice is deduped to one input, and both
// loads read the surviving arg (foreach analog of the opaque case in
// kernel_operands_basic.mlir).
// CHECK-LABEL: mo.graph @dedupes_repeated_foreach_input
// CHECK:       map.iter.elementwise<[4]>
// CHECK-NEXT:  inputs ([[A:%[^ :]+]]: !mo.tensor<[4], f32>)
// CHECK-NEXT:  outputs
// CHECK:         map.load [[A]][
// CHECK:         map.load [[A]][
mo.graph @dedupes_repeated_foreach_input(
  %x: !mo.tensor<[4], f32>
) -> !mo.tensor<[4], f32> {
  %y = map.iter.elementwise<[4]>
      inputs  (%x as %a: !mo.tensor<[4], f32>, %x as %b: !mo.tensor<[4], f32>)
      outputs (%out: !mo.tensor<[4], f32>)
      index   (%i: !map.index<1>)
      -> !mo.tensor<[4], f32>
  {
    %av = map.load %a[%i] : !mo.tensor<[4], f32>
    %bv = map.load %b[%i] : !mo.tensor<[4], f32>
    %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%i, %av, %bv)
               : !map.index<1>, f32, f32 -> f32
    map.store %out[%i] = %sum : !mo.tensor<[4], f32>
  }
  mo.output %y : !mo.tensor<[4], f32>
}
}
