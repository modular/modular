// RUN: mo-opt %s --gc-pipeline-pre-jit="kernel-library-paths=%kernel_api_test_pkg dump-mojo-text=true" 2>&1 | FileCheck %s

// A throwing shape function's stub returns a value through the try, so its
// result variable is declared with the result's own type. Declaring it as an
// MGP async value -- the only shape this emitter had -- does not compile.

// CHECK: var [[VAR:[a-z_0-9]+]]: std.utils.IndexList[
// CHECK:   try:
// CHECK:     [[VAR]] =
// CHECK:   except
// CHECK:     raise
// CHECK:   return [[VAR]]

mo.graph @raising_shape<D0 -> D1>(%a: !mo.tensor<[D0], f32>, %b: !mo.tensor<[D0], f32>) -> !mo.tensor<[D1], f32> {
  %r = mo.custom<() -> D2> [symbol = "imposter_add_raising_shape", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D2], f32>
  mo.output<D2> %r : !mo.tensor<[D2], f32>
}
