// RUN: mo-opt %s --map-fuse-reductions | FileCheck %s

// Fusing a rowwise kernel through a cache, on rows written as MAP IR.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Mul registration = "mo.mul" elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// The later kernel reads two of the map phase's outputs, but a cache holds one
// value: the kernels stay apart.
// CHECK-LABEL: mo.graph @twoOutputsRead
// CHECK-COUNT-2: map.iter.rowwise
// CHECK-NOT:     map.iter.cache
mo.graph @twoOutputsRead(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %hg:2 = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%h: !mo.tensor<[8, 128], f32>, %g: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8, 128], f32>) {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
    map.iter.map index (%j: !map.index<2>) {
      %v = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %hv = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %v, %acc) : !map.index<2>, f32, f32 -> f32
      map.store %h[%j] = %hv : !mo.tensor<[8, 128], f32>
      map.store %g[%j] = %v : !mo.tensor<[8, 128], f32>
    }
  }
  %y = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%hg#0: !mo.tensor<[8, 128], f32>, %hg#1: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %hv = map.load %hg#0[%i] : !mo.tensor<[8, 128], f32>
      map.yield %hv : f32
    }
    map.iter.map index (%j: !map.index<2>) {
      %hv = map.load %hg#0[%j] : !mo.tensor<[8, 128], f32>
      %gv = map.load %hg#1[%j] : !mo.tensor<[8, 128], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %hv, %gv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// One output read at the same element: one kernel, the first kernel's map
// phase a cache the later phases read. The output no one else reads isn't
// stored.
// CHECK-LABEL: mo.graph @oneOutputRead
// CHECK:         map.iter.rowwise
// CHECK:         %[[C:.*]] = map.iter.cache
// CHECK-NOT:       map.store
// CHECK:           map.yield
// CHECK:         map.iter.cache.load %[[C]]
// CHECK:         map.iter.cache.load %[[C]]
// CHECK-NOT:   map.iter.rowwise
mo.graph @oneOutputRead(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %h = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%ho: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
    map.iter.map index (%j: !map.index<2>) {
      %v = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %hv = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %v, %acc) : !map.index<2>, f32, f32 -> f32
      map.store %ho[%j] = %hv : !mo.tensor<[8, 128], f32>
    }
  }
  %y = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%h: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %hv = map.load %h[%i] : !mo.tensor<[8, 128], f32>
      map.yield %hv : f32
    }
    map.iter.map index (%j: !map.index<2>) {
      %hv = map.load %h[%j] : !mo.tensor<[8, 128], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %hv, %acc) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %y : !mo.tensor<[8, 128], f32>
}
