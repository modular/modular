// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s

// A foreach's width limit reaches `_foreach` as its `max_simd_width`: an
// `Idx` when static, else the run-time largest power of two dividing the
// limit's terms, static part included.

// CHECK:      var [[DIV:var_+[0-9]+]] = builtin_primitives.primitives.pow2_divisor_of_dims(std.utils.coord.coord_to_index_list({{var_+[0-9]+}}))
// CHECK-NEXT: builtin_primitives.primitives._foreach[simd_width = {{.*}}]({{.*}}, max_simd_width = Scalar[DType.int]([[DIV]]))
mo.graph @dynamic<N>(%a: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32>)
      outputs (%out : !mo.tensor<[N], f32>) {
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[8, N]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32>, %out0 = %out : !mo.tensor<[N], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[N], f32>
}

// CHECK:      builtin_primitives.primitives._foreach[simd_width = {{.*}}]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, max_simd_width = layout.Idx[2])
mo.graph @static(%a: !mo.tensor<[16], f32>) -> !mo.tensor<[16], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[16], f32>)
      outputs (%out : !mo.tensor<[16], f32>) {
    map.iter.elementwise.body<[16]> max_simd_width(#map.gcd<[6, 10]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[16], f32>, %out0 = %out : !mo.tensor<[16], f32>} {
      %v = map.load %a0[%idx] : !mo.tensor<[16], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[16], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[16], f32>
}
