// RUN: mo-opt %s --mogg-import-packages --map-to-mogg | FileCheck %s

// A `map.iter.elementwise.body`'s `max_simd_width`, simplified (nested GCDs
// flattened, constants folded), lowers on `mogg.call.foreach` to a number when
// it is one, else to a shape of its terms (the folded constant, then each
// symbolic dim), which gives each dim its run-time `!mogg.dim`.

// CHECK-LABEL: mo.graph @dynamic
// CHECK:       dims ([[DN:%[^ :,]+]]: !mogg.dim<N>)
// CHECK:       mogg.tensor.create({{.*}} io = Output
// CHECK-NEXT:  mogg.shape.create([[DN]])
// CHECK-NEXT:  [[WSH:%[^ ]+]] = mogg.shape.create([[DN]]) <shape = #mosh<ape[2, dyn]> : !mosh.ape>
// CHECK-NEXT:  mogg.call.foreach @builtin_primitives.primitives._foreach shape({{%[^ ]+}}) context({{%[^ ]+}}) dynamic_max_simd_width([[WSH]])
// CHECK-SAME:  device = "cpu" {
mo.graph @dynamic<N>(%a: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32>)
      outputs (%out : !mo.tensor<[N], f32>) {
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[#map.gcd<[6, N]>, 10, N]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[N], f32>
}

// CHECK-LABEL: mo.graph @static
// CHECK:       mogg.call.foreach @builtin_primitives.primitives._foreach shape({{%[^ ]+}}) context({{%[^ ]+}}) (
// CHECK-SAME:  device = "cpu" attributes {max_simd_width = 2 : i64}
mo.graph @static(%a: !mo.tensor<[16], f32>) -> !mo.tensor<[16], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[16], f32>)
      outputs (%out : !mo.tensor<[16], f32>) {
    map.iter.elementwise.body<[16]> max_simd_width(#map.gcd<[#map.gcd<[6, 12]>, 10]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[16], f32>, %out0 = %out : !mo.tensor<[16], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[16], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[16], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[16], f32>
}
