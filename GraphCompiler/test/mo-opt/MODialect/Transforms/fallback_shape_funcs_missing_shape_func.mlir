// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-add-fallback-shape-funcs --verify-diagnostics

// `print_tensor` has no shape function, so nothing can resolve D1. The pass
// should say so rather than call a shape function that isn't there.

mo.graph @missing_shape_func<D0 -> O0>(%input: !mo.tensor<[D0], f32>) -> !mo.tensor<[O0], f32> {
  // expected-error @+1 {{custom op 'print_tensor' with data-dependent output dimensions requires a registered shape function}}
  %res = mo.custom<() -> D1> [symbol = "print_tensor", device = #M.device_ref<"cpu", 0>](%input) : (!mo.tensor<[D0], f32>) -> !mo.tensor<[D1], f32>
  mo.output<D1> %res : !mo.tensor<[D1], f32>
}
