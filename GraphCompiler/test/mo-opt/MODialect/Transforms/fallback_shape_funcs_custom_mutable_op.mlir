// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-add-fallback-shape-funcs | FileCheck %s

mo.graph @dyn_shape_func<D0, D1 -> D4, D5>(%arg0: !mo.buffer<[D0, D1], f32>) -> (!mo.buffer<[D0, D1], f32>, !mo.tensor<[D4, D5], f32>, !mo.chain) {
  %0 = mo.chain.create()
  %1:2 = mo.custom<() -> D2, D3> [symbol = "shape_func_with_mutable"](%arg0, %0) : (!mo.buffer<[D0, D1], f32>, !mo.chain) -> (!mo.tensor<[D2, D3], f32>, !mo.chain)
  mo.output<D2, D3> %arg0, %1#0, %1#1 : !mo.buffer<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.chain
}

// CHECK-LABEL: @dyn_shape_func
// CHECK-NEXT: %[[C0:.*]] = mo.chain.create()
// CHECK-NEXT: %[[T:.*]], %[[C1:.*]] = mo.mutable.load(%arg0, %[[C0]])
// CHECK-NEXT: mo.invoke_shape_func["shape_func_with_mutable"]<{{.*}}>(%[[T]])
// CHECK-NEXT: mo.custom{{.*}}%[[C1]]
