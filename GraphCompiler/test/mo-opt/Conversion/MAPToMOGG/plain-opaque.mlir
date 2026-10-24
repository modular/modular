// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg | FileCheck %s

// MAPToMOGG lowers a plain-output `map.iter.opaque` into a `mogg.routine`:
// inputs stay `!mo.tensor` operands and outputs stay `!mo.tensor` results, but
// inside the body each tensor is reconstructed from a `!mogg.pointer` block
// argument via `mogg.shape.create` / `mogg.tensor.create` (each recording the
// kernel's `device`), and the kernel call becomes `mogg.call.opaque` wired to
// the body's trailing `!mogg.context`. The grouped `_kernel` header lists the
// pointer block args as `inputs(...) outputs(...) context(...)`. (Store-epilogue
// / fused outputs are deferred and not exercised here.)

// Single plain output.
// CHECK-LABEL: mo.graph @one_plain
// CHECK: mogg.routine (%[[A:.*]] : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: inputs (%[[IP:[^ ,)]+]])
// CHECK: outputs (%[[OP:[^ ,)]+]])
// CHECK: context (%[[CTX:[^ ,)]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:   %[[ISH:.*]] = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
// CHECK:   %[[IT:.*]] = mogg.tensor.create(%[[IP]], %[[ISH]]) <device = "cpu",{{.*}}mut = false{{.*}}> : !mogg.tensor
// CHECK:   %[[OSH:.*]] = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
// CHECK:   %[[OT:.*]] = mogg.tensor.create(%[[OP]], %[[OSH]]) <device = "cpu",{{.*}}mut = true{{.*}}> : !mogg.tensor
// CHECK:   mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%[[IT]] : !mogg.tensor) outputs(%[[OT]] : !mogg.tensor) context(%[[CTX]] : !mogg.context)
// Nothing is fused here, so the epilogue flag keeps its `false` default, which
// the printer elides -- its absence is the assertion.
// CHECK-NOT: hasEpilogueFusion
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two plain outputs: one input pointer, two output pointers, one context.
// CHECK-LABEL: mo.graph @two_plain
// CHECK: mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK: inputs (%{{.*}})
// CHECK: outputs (%{{.*}}, %{{.*}})
// CHECK: context (%[[CTX:[^ ,)]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:   mogg.call.opaque @{{.*}}MapLowerTwoPlainOutputs inputs(%{{.*}} : !mogg.tensor) outputs(%{{.*}}, %{{.*}} : !mogg.tensor, !mogg.tensor) context(%[[CTX]] : !mogg.context)
mo.graph @two_plain(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_two_plain_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
