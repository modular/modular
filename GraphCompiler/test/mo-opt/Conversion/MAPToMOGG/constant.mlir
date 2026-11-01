// RUN: mo-opt %s --mogg-import-packages --map-to-mogg | FileCheck %s

// A `map.constant` has no operand to arrive on, so it lowers to a
// `mogg.buffer.constant` -- a pointer straight from the bytes -- and its tensor
// is rebuilt from that pointer exactly like an input's. Note the kernel takes
// no input pointer at all here: the constant is not an operand.
//
// The bytes stay unaligned. `MOGGToMGP` turns this into an
// `mgp.buffer.constant` at the call site, where the pipeline's minimum CPU
// alignment is known.

// CHECK-LABEL: mo.graph @constant
// CHECK: mogg.routine () -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT: outputs ([[OP:%[^ :,]+]])
// CHECK-NEXT: context ([[CTX:%[^ :,]+]])
// CHECK-NEXT:   [[OSH:%[^ ]+]] = mogg.shape.create()
// CHECK-NEXT:   [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) {{.*}}io = Output, mut = true
// CHECK-NEXT:   [[CP:%[^ ]+]] = mogg.buffer.constant #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:   [[CSH:%[^ ]+]] = mogg.shape.create()
// CHECK-NEXT:   [[CT:%[^ ]+]] = mogg.tensor.create([[CP]], [[CSH]]) {{.*}}io = Input, mut = false
// CHECK-NEXT:   [[LSH:%[^ ]+]] = mogg.shape.create()
// CHECK-NEXT:   mogg.call.foreach {{.*}} ([[IDX:%[^ :,]+]] : !map.index<2>) capturing {[[CIN:%[^ :,]+]] = [[CT]] : !mogg.tensor, [[OUT:%[^ :,]+]] = [[OT]] : !mogg.tensor}
// CHECK-NEXT:     [[CV:%[^ ]+]] = mogg.tensor.load [[CIN]][[[IDX]]]
// CHECK-NEXT:     mogg.tensor.store [[OUT]][[[IDX]]] = [[CV]]
mo.graph @constant() -> !mo.tensor<[2, 2], f32> {
  %r = map.iter.opaque
      inputs ()
      outputs (%out : !mo.tensor<[2, 2], f32>) {
    %c = map.constant #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32> : !mo.tensor<[2, 2], f32>
    map.iter.elementwise.body<[2, 2]>
        index (%idx: !map.index<2>)
        capturing {%c0 = %c : !mo.tensor<[2, 2], f32>, %out0 = %out : !mo.tensor<[2, 2], f32>} {
      %cv = map.load %c0[%idx] : !mo.tensor<[2, 2], f32>
      map.store %out0[%idx] = %cv : !mo.tensor<[2, 2], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[2, 2], f32>
}
