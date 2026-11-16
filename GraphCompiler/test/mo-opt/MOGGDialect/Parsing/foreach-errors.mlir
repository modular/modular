// RUN: mo-opt %s -split-input-file -verify-diagnostics

func.func @staticAndDynamicLimit(%shape: !mogg.shape, %ctx: !mogg.context, %limit: !mogg.shape, %x: !mogg.tensor, %out: !mogg.tensor) {
  // expected-error @below{{expected a static or a dynamic max_simd_width, not both}}
  mogg.call.foreach @builtin_primitives.primitives._foreach shape(%shape) context(%ctx) dynamic_max_simd_width(%limit) (%idx : !map.index<1>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" attributes {max_simd_width = 2 : i64} {
    %v = mogg.tensor.load %cx[%idx] : !mogg.tensor, !map.index<1> -> !mogg.simd<f32>
    mogg.tensor.store %co[%idx] = %v : !mogg.tensor, !map.index<1>, !mogg.simd<f32>
  }
  return
}
