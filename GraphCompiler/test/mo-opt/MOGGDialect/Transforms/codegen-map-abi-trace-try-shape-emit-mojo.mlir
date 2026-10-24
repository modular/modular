// RUN: mo-opt %s --gc-pipeline-pre-jit="kernel-library-paths=%kernel_api_test_pkg dump-mojo-text=true" 2>&1 | FileCheck %s

// A throwing shape function gets both wrappers and still returns: each
// forwards the shape out, so it survives both hops. Without that forwarding
// the shape is computed inside a region and discarded.

// CHECK:       def __shapeof_imposter_add_raising_shape{{[_0-9]*}}(
// CHECK-SAME:    -> std.utils.IndexList[
// CHECK:         var [[OUT:[a-z_0-9]+]]: std.utils.IndexList[
// CHECK:         try:
// CHECK:           var [[TRACED:[a-z_0-9]+]]: std.utils.IndexList[
// CHECK:           with max.runtime.tracing.Trace[{{.*}}]("shapeof(imposter_add_raising_shape)"
// CHECK:             var [[VAL:[a-z_0-9]+]] = {{.*}}imposter_add_raising_shape_shape(
// CHECK:             [[TRACED]] = [[VAL]]
// CHECK:           [[OUT]] = [[TRACED]]
// CHECK:         except
// CHECK:           raise
// CHECK:         return [[OUT]]

mo.graph @raising_shape<D0 -> D1>(%a: !mo.tensor<[D0], f32>, %b: !mo.tensor<[D0], f32>) -> !mo.tensor<[D1], f32> {
  %r = mo.custom<() -> D2> [symbol = "imposter_add_raising_shape", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D2], f32>
  mo.output<D2> %r : !mo.tensor<[D2], f32>
}
