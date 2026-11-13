// RUN: mo-opt %s --split-input-file --verify-diagnostics

// A symbolic dim in a limit must be a parameter: the lowering finds its value
// among the kernel's dim arguments, which hold no expressions.

mo.graph @expression_in_gcd<N>(%a: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32>)
      outputs (%out : !mo.tensor<[N], f32>) {
    // expected-error @below{{expected a constant or a parameter dim, got}}
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[16, add(N, 8)]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[N], f32>
}

// -----

mo.graph @expression<N>(%a: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32>)
      outputs (%out : !mo.tensor<[N], f32>) {
    // expected-error @below{{expected a constant or a parameter dim, got}}
    map.iter.elementwise.body<[N]> max_simd_width(add(N, 8))
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[N], f32>
}
