// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg | FileCheck %s

// Dynamic shapes: `mogg.routine` gains one `!mogg.dim` block argument per UNIQUE
// dynamic dimension (deduped across inputs and outputs), listed in the grouped
// header's `dims(...)` after `context(...)`, and each `mogg.shape.create` is fed
// the dims for its own tensor. The dim block arg keeps its source parameter name
// (`!mogg.dim<batch>`), but in the `shape.create` shape ATTR every dynamic dim is
// canonicalized to a single `dyn` placeholder (statics stay literal) so shapes
// that differ only in dim symbols dedup. Here the input and output share the
// dynamic dim `batch`, so there is exactly one `!mogg.dim<batch>` block arg and
// both `shape.create`s use it, each printing `#mosh<ape[dyn, 4]>`.

// CHECK-LABEL: mo.graph @dyn
// CHECK: mogg.routine (%{{.*}} : !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
// CHECK: inputs (%{{.*}})
// CHECK: outputs (%{{.*}})
// CHECK: context (%[[CTX:[^ ,)]+]])
// CHECK: dims (%[[D:.*]]: !mogg.dim<batch>) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK: mogg.shape.create(%[[D]]) <shape = #mosh<ape[dyn, 4]> : !mosh.ape> : !mogg.dim<batch>
// CHECK: mogg.shape.create(%[[D]]) <shape = #mosh<ape[dyn, 4]> : !mosh.ape> : !mogg.dim<batch>
// CHECK: mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%{{.*}} : !mogg.tensor) outputs(%{{.*}} : !mogg.tensor) context(%[[CTX]] : !mogg.context)
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}
