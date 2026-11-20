// RUN: mo-opt %s | mo-opt | FileCheck %s

// A SIMD width limit is a constant, a parameter dim, or a `#map.gcd<[...]>` of
// limits, which nests. It prints as written: only the pass and the lowering
// simplify it.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

// map.iter.reduce verifies its monoid resolves to a mogg.kernel.decl.
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: mo.graph @limits
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(8)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(N)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[6, 10]>)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[N, 16, N]>)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[#map.gcd<[N, 6]>, 10]>)
// CHECK: map.iter.rowwise.body<[N, 32], axis = 1> max_simd_width(#map.gcd<[4, N]>)
mo.graph @limits<N>(%a: !mo.tensor<[N], f32>, %x: !mo.tensor<[N, 32], f32>) -> (!mo.tensor<[N], f32>, !mo.tensor<[N, 1], f32>) {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32>)
      outputs (%out : !mo.tensor<[N], f32>) {
    map.iter.elementwise.body<[N]> max_simd_width(8)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
    map.iter.elementwise.body<[N]> max_simd_width(N)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[6, 10]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[N, 16, N]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[#map.gcd<[N, 6]>, 10]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  %s = map.iter.opaque
      inputs (%x : !mo.tensor<[N, 32], f32>)
      outputs (%sum : !mo.tensor<[N, 1], f32>) {
    map.iter.rowwise.body<[N, 32], axis = 1> max_simd_width(#map.gcd<[4, N]>)
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[N, 32], f32>, %sum0 = %sum : !mo.tensor<[N, 1], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[N, 32], f32>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[N, 1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r, %s : !mo.tensor<[N], f32>, !mo.tensor<[N, 1], f32>
}
