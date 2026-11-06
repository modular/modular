// RUN: mo-opt %s --map-rcse | FileCheck %s

// map-rcse on rows written as MAP IR, with shapes fusion doesn't produce yet:
// two folds, and a phase that computes the value twice.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Mul registration = "mo.mul" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// A phase that computes x * y twice reads the cache twice.
// CHECK-LABEL: mo.graph @twoCopies
// CHECK:         %[[P:.*]] = map.iter.cache
// CHECK:         map.iter.map index (%[[J:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[A:.*]] = map.iter.cache.load %[[P]][%[[J]]]
// CHECK-NEXT:      %[[B:.*]] = map.iter.cache.load %[[P]][%[[J]]]
// CHECK-NEXT:      %[[R:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[J]], %[[A]], %[[B]])
// CHECK-NEXT:      map.store %{{.*}}[%[[J]]] = %[[R]]
mo.graph @twoCopies(%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %yv = map.load %y[%i] : !mo.tensor<[8, 128], f32>
      %q = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %yv) : !map.index<2>, f32, f32 -> f32
      map.yield %q : f32
    }
    map.iter.map index (%j: !map.index<2>) {
      %xa = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %ya = map.load %y[%j] : !mo.tensor<[8, 128], f32>
      %qa = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xa, %ya) : !map.index<2>, f32, f32 -> f32
      %xb = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %yb = map.load %y[%j] : !mo.tensor<[8, 128], f32>
      %qb = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xb, %yb) : !map.index<2>, f32, f32 -> f32
      %rv = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %qa, %qb) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %rv : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// Two folds and a map phase all compute x * y: each reads the cache.
// CHECK-LABEL: mo.graph @twoFoldsAndMap
// CHECK:         %[[P:.*]] = map.iter.cache index (%[[K:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[X:.*]] = map.load %{{.*}}[%[[K]]]
// CHECK-NEXT:      %[[Y:.*]] = map.load %{{.*}}[%[[K]]]
// CHECK-NEXT:      %[[Q:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[K]], %[[X]], %[[Y]])
// CHECK-NEXT:      map.yield %[[Q]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%[[I1:.*]]: !map.index<2>)
// CHECK-NEXT:      %[[V1:.*]] = map.iter.cache.load %[[P]][%[[I1]]]
// CHECK-NEXT:      map.yield %[[V1]]
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%[[I2:.*]]: !map.index<2>)
// CHECK-NEXT:      %[[V2:.*]] = map.iter.cache.load %[[P]][%[[I2]]]
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I2]], %[[V2]],
// CHECK:         map.iter.map index (%[[J:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[W:.*]] = map.iter.cache.load %[[P]][%[[J]]]
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Mul(%[[J]], %[[W]],
// CHECK-NOT:     map.load %{{.*}}[%[[J]]]
mo.graph @twoFoldsAndMap(%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %y: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %a = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %yv = map.load %y[%i] : !mo.tensor<[8, 128], f32>
      %q = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %yv) : !map.index<2>, f32, f32 -> f32
      map.yield %q : f32
    }
    %b = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %yv = map.load %y[%i] : !mo.tensor<[8, 128], f32>
      %q = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %yv) : !map.index<2>, f32, f32 -> f32
      %d = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %q, %a) : !map.index<2>, f32, f32 -> f32
      map.yield %d : f32
    }
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %yv = map.load %y[%j] : !mo.tensor<[8, 128], f32>
      %q = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %yv) : !map.index<2>, f32, f32 -> f32
      %rv = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %q, %b) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %rv : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}
