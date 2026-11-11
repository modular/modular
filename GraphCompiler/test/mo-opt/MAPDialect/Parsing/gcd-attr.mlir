// RUN: mo-opt %s | mo-opt | FileCheck %s

// A SIMD width limit is a constant, a parameter dim, or a `#map.gcd<[...]>` of
// limits, which nests. It prints as written: only the pass and the lowering
// simplify it.

// CHECK-LABEL: mo.graph @limits
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(8)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(N)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[6, 10]>)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[N, 16, N]>)
// CHECK: map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[#map.gcd<[N, 6]>, 10]>)
mo.graph @limits<N>(%a: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
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
  mo.output %r : !mo.tensor<[N], f32>
}
