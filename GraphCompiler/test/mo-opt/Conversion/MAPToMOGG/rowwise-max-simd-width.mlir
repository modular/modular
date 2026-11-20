// RUN: mo-opt %s --mogg-import-packages --map-to-mogg | FileCheck %s

// A `map.iter.rowwise.body`'s `max_simd_width`, simplified (nested GCDs
// flattened, constants folded), lowers on `mogg.call.rowwise` as on
// `mogg.call.foreach`: to a number when it is one, else to a shape of its
// terms (the folded constant, then each symbolic dim).

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

// map.iter.reduce verifies its monoid resolves to a mogg.kernel.decl.
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: mo.graph @dynamic
// CHECK:       dims ([[DN:%[^ :,]+]]: !mogg.dim<N>)
// CHECK:       [[WSH:%[^ ]+]] = mogg.shape.create([[DN]]) <shape = #mosh<ape[2, dyn]> : !mosh.ape>
// CHECK-NEXT:  mogg.call.rowwise shape({{%[^ ]+}}) axis(1) context({{%[^ ]+}}) dynamic_max_simd_width([[WSH]])
// CHECK-SAME:  device = "cpu" {
mo.graph @dynamic<N>(%x: !mo.tensor<[N, 32], f32>) -> !mo.tensor<[N, 1], f32> {
  %s = map.iter.opaque
      inputs (%x : !mo.tensor<[N, 32], f32>)
      outputs (%sum : !mo.tensor<[N, 1], f32>) {
    map.iter.rowwise.body<[N, 32], axis = 1> max_simd_width(#map.gcd<[#map.gcd<[6, N]>, 10, N]>)
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[N, 32], f32>, %sum0 = %sum : !mo.tensor<[N, 1], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[N, 32], f32>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[N, 1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %s : !mo.tensor<[N, 1], f32>
}

// CHECK-LABEL: mo.graph @static
// CHECK:       mogg.call.rowwise shape({{%[^ ]+}}) axis(1) context({{%[^ ]+}}) (
// CHECK-SAME:  device = "cpu" attributes {max_simd_width = 2 : i64}
mo.graph @static(%x: !mo.tensor<[4, 32], f32>) -> !mo.tensor<[4, 1], f32> {
  %s = map.iter.opaque
      inputs (%x : !mo.tensor<[4, 32], f32>)
      outputs (%sum : !mo.tensor<[4, 1], f32>) {
    map.iter.rowwise.body<[4, 32], axis = 1> max_simd_width(#map.gcd<[#map.gcd<[6, 12]>, 10]>)
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[4, 32], f32>, %sum0 = %sum : !mo.tensor<[4, 1], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[4, 32], f32>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[4, 1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %s : !mo.tensor<[4, 1], f32>
}
