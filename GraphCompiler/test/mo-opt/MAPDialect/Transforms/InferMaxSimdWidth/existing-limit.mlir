// RUN: mo-opt %s --map-infer-max-simd-width | FileCheck %s
// RUN: mo-opt %s --map-infer-max-simd-width --map-infer-max-simd-width | FileCheck %s

// A limit already on a body constrains its width too: the pass takes the GCD
// with what the body's ops need, so running it again keeps the limit.

// CHECK-LABEL: mo.graph @static
// CHECK:       map.iter.elementwise.body<[16]> max_simd_width(6)
mo.graph @static(%a: !mo.tensor<[16], f32>) -> !mo.tensor<[16], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[16], f32>)
      outputs (%out : !mo.tensor<[16], f32>) {
    map.iter.elementwise.body<[16]> max_simd_width(6)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[16], f32>, %out0 = %out : !mo.tensor<[16], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[16], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[16], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[16], f32>
}

// CHECK-LABEL: mo.graph @dynamic
// CHECK:       map.iter.elementwise.body<[N]> max_simd_width(N)
mo.graph @dynamic<N>(%a: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32>)
      outputs (%out : !mo.tensor<[N], f32>) {
    map.iter.elementwise.body<[N]> max_simd_width(N)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[N], f32>
}
