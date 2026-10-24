// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s

// MAPToMOGG lowers a store-epilogue (fused) output of a `map.iter.opaque`
// directly into a `mogg.tensor.create.epilogue` holding the
// `mogg.tensor.store` in its own body (capturing the raw output tensor); the
// kernel call writes into the epilogue tensor rather than the raw output
// tensor. The epilogue tensor reuses the store target's `!mogg.shape` operand
// and records the kernel `device`, and the call is wired to the body's
// `!mogg.context`.
//
// The first four graphs fuse nothing INTO their epilogue, so the call carries
// no `hasEpilogueFusion`: the flag tracks whether the body recomputes the
// value, not whether an epilogue exists. Every fused-output kernel gets one.
// The last three are the other polarity.

// One fused output.
// CHECK-LABEL: mo.graph @one_fused
// CHECK: mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: context (%[[CTX:[^ ,)]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK: %[[OT:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) <device = "cpu",{{.*}}mut = true{{.*}}> : !mogg.tensor
// CHECK: %[[ESH:.*]] = mogg.shape.create()
// CHECK: %[[EPI:[^ ,)]+]] = mogg.tensor.create.epilogue %[[ESH]] (%[[V:[^ ,)]+]] : !mogg.simd<f32>, %[[I:[^ ,)]+]] : !map.index<2>) capturing {%[[CAP:[^ ,)]+]] = %[[OT]] : !mogg.tensor} ptr %{{.*}} device = "cpu",{{.*}}mut = true {
// CHECK:   mogg.tensor.store %[[CAP]][%[[I]]] = %[[V]] : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
// CHECK: mogg.call.opaque @{{.*}}MapLowerOneFusedOutput inputs(%{{.*}} : !mogg.tensor) outputs(%[[EPI]] : !mogg.tensor) context(%[[CTX]] : !mogg.context)
// CHECK-NOT: hasEpilogueFusion
mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two fused outputs: two epilogue tensors, both written by the call.
// CHECK-LABEL: mo.graph @two_fused
// CHECK: %[[OT0:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) <device = "cpu",{{.*}}mut = true{{.*}}> : !mogg.tensor
// CHECK: %[[OT1:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) <device = "cpu",{{.*}}mut = true{{.*}}> : !mogg.tensor
// CHECK: %[[ESH0:.*]] = mogg.shape.create()
// CHECK: %[[E0:.*]] = mogg.tensor.create.epilogue %[[ESH0]] ({{[^)]*}}) capturing {%{{[^ ]+}} = %[[OT0]] : !mogg.tensor} ptr %{{.*}} device = "cpu",{{.*}}mut = true {
// CHECK: %[[ESH1:.*]] = mogg.shape.create()
// CHECK: %[[E1:.*]] = mogg.tensor.create.epilogue %[[ESH1]] ({{[^)]*}}) capturing {%{{[^ ]+}} = %[[OT1]] : !mogg.tensor} ptr %{{.*}} device = "cpu",{{.*}}mut = true {
// CHECK: mogg.call.opaque @{{.*}}MapLowerTwoFusedOutputs inputs(%{{.*}} : !mogg.tensor) outputs(%[[E0]], %[[E1]] : !mogg.tensor, !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK-NOT: hasEpilogueFusion
mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_two_fused_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Two inputs feeding one fused output: the epilogue's captured output block
// arg number is offset by the two leading input block args, so this exercises
// that the capture still resolves to the right (only) output tensor.
// CHECK-LABEL: mo.graph @two_inputs_one_fused
// CHECK: mogg.routine (%{{.*}}, %{{.*}} : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: %[[OT:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) <device = "cpu",{{.*}}mut = true{{.*}}> : !mogg.tensor
// CHECK: %[[ESH:.*]] = mogg.shape.create()
// CHECK: %[[EPI:.*]] = mogg.tensor.create.epilogue %[[ESH]] ({{[^)]*}}) capturing {%{{[^ ]+}} = %[[OT]] : !mogg.tensor} ptr %{{.*}} device = "cpu",{{.*}}mut = true {
// CHECK: mogg.call.opaque @{{.*}}MapLowerTwoInputsOneFusedOutput inputs(%{{.*}}, %{{.*}} : !mogg.tensor, !mogg.tensor) outputs(%[[EPI]] : !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK-NOT: hasEpilogueFusion
mo.graph @two_inputs_one_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_two_inputs_one_fused_output", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Mixed: first output fused (epilogue tensor), second plain (raw output tensor).
// CHECK-LABEL: mo.graph @two_partial
// CHECK: %[[EPI:.*]] = mogg.tensor.create.epilogue %{{.*}} ({{[^)]*}}) capturing {%{{[^ ]+}} = %{{.*}} : !mogg.tensor} ptr %{{.*}} device = "cpu",{{.*}}mut = true {
// CHECK: mogg.call.opaque @{{.*}}MapLowerPartialFusedOutputs inputs(%{{.*}} : !mogg.tensor) outputs(%[[EPI]], %{{.*}} : !mogg.tensor, !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK-NOT: hasEpilogueFusion
mo.graph @two_partial(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_partial_fused_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A consumer fused into the epilogue: its body recomputes the value, so the
// flag is set. The mirror of @fused_producer in fused-input-opaque.mlir.

// CHECK-LABEL: mo.graph @fused_consumer
// CHECK: mogg.call.elementwise @
// CHECK: mogg.call.opaque @{{.*}}MapLowerOneFusedOutput {{.*}}hasEpilogueFusion = true
mo.graph @fused_consumer(%a: !mo.tensor<[2, 2], f32>, %z: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.add(%c, %z) : !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// A pure reshape stays its own view kernel rather than moving onto the store
// index, so the epilogue body is still a bare store.

// CHECK-LABEL: mo.graph @reshape_only
// CHECK-NOT: hasEpilogueFusion
mo.graph @reshape_only(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}

// A reshape between the kernel and an add does move onto the store index, so
// this body remaps the index AND recomputes the value. Only the second counts,
// which is why the flag is read off the value and not the body's size.

// CHECK-LABEL: mo.graph @reshape_and_add
// CHECK: mogg.index.reshape
// CHECK: mogg.call.elementwise @
// CHECK: mogg.call.opaque @{{.*}}MapLowerOneFusedOutput {{.*}}hasEpilogueFusion = true
mo.graph @reshape_and_add(%a: !mo.tensor<[2, 2], f32>, %z: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  %d = mo.add(%r, %z) : !mo.tensor<[4], f32>
  mo.output %d : !mo.tensor<[4], f32>
}
