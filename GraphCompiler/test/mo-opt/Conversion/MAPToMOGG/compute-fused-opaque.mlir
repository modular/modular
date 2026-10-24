// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg="enable-compute-lambdas=false" | FileCheck %s --check-prefix=NOCOMPUTE

// The GPU matmul declares its output as `_FusedComputeOutputTensor`, so an
// epilogue fused into it can be a COMPUTE lambda: the body yields the value the
// store would have written and the kernel performs the one store itself. The
// store-target capture (`%[[COUT]]`) survives unused -- emit still binds the
// functor onto its slice.
// CHECK-LABEL: mo.graph @matmul_add
// CHECK:       mogg.tensor.create.epilogue %{{[^ ]+}} (%[[VAL:[^ ]+]] : !mogg.simd<f32>, %[[IDX:[^ ]+]] : !map.index<2>) capturing {%[[COUT:[^ ]+]] = %{{[^ ]+}} : !mogg.tensor, %[[CZ:[^ ]+]] = %{{[^ ]+}} : !mogg.tensor} {{.*}}computeOutputArg = true
// CHECK-NEXT:    %[[ZV:.*]] = mogg.tensor.load %[[CZ]][%[[IDX]]]
// CHECK-NEXT:    %[[SUM:.*]] = mogg.call.elementwise @{{.*}}Add{{.*}}(%[[VAL]], %[[ZV]])
// CHECK-NEXT:    mogg.yield %[[SUM]] : !mogg.simd<f32>

// With compute lambdas off the same graph keeps a classic store lambda, but
// `computeOutputArg` still records what the kernel argument declares -- emit
// must bind through the compute entry point either way.
// NOCOMPUTE-LABEL: mo.graph @matmul_add
// NOCOMPUTE:       mogg.tensor.create.epilogue %{{[^ ]+}} (%{{[^ ]+}} : !mogg.simd<f32>, %[[IDX:[^ ,)]+]] : !map.index<2>) capturing {%[[COUT:[^ ,)]+]] = {{.*}}computeOutputArg = true
// NOCOMPUTE-NEXT:    mogg.tensor.load
// NOCOMPUTE-NEXT:    %[[SUM:.*]] = mogg.call.elementwise
// NOCOMPUTE-NEXT:    mogg.tensor.store %[[COUT]][%[[IDX]]] = %[[SUM]]
mo.graph @matmul_add(
  %a: !mo.tensor<[8, 8], f32, gpu:0>,
  %b: !mo.tensor<[8, 8], f32, gpu:0>,
  %z: !mo.tensor<[8, 8], f32, gpu:0>
) -> !mo.tensor<[8, 8], f32, gpu:0> {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[8, 8], f32, gpu:0>, !mo.tensor<[8, 8], f32, gpu:0>) -> !mo.tensor<[8, 8], f32, gpu:0>
  %out = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32, gpu:0>
  mo.output %out : !mo.tensor<[8, 8], f32, gpu:0>
}

// A dtype-changing epilogue keeps a store lambda even though the kernel could
// take a compute one: `compute()` returns the dtype it was handed, so a body
// whose value leaves as si32 having arrived as f32 cannot round-trip through
// it. `computeOutputArg` is still set, because the kernel argument is still a
// `_FusedComputeOutputTensor`.
// CHECK-LABEL: mo.graph @mixed_dtype
// CHECK:       mogg.tensor.create.epilogue %{{[^ ]+}} (%[[V:[^ ]+]] : !mogg.simd<f32>, %[[I:[^ ]+]] : !map.index<2>) capturing {%[[CO:[^ ]+]] = {{.*}}computeOutputArg = true
// CHECK:         mogg.tensor.store %[[CO]][%[[I]]]
mo.graph @mixed_dtype(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>,
  %bias: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], si32> {
  %sum = mo.custom [symbol = "imposter_add_compute_fusion", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %added = mo.add(%sum, %bias) : !mo.tensor<[2, 2], f32>
  %out = mo.cast(%added) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  mo.output %out : !mo.tensor<[2, 2], si32>
}

// A store-only kernel argument is untouched: store lambda, and the attribute
// stays at its `false` default, which the printer elides. `computeOutputArg`
// sorts before `device`, so `device` following the output pointer immediately
// proves it is absent.
// CHECK-LABEL: mo.graph @plain_store
// CHECK:       mogg.tensor.create.epilogue %{{[^ ]+}} (%[[PV:[^ ]+]] : !mogg.simd<f32>, %[[PI:[^ ]+]] : !map.index<1>) capturing {%[[PO:[^ ]+]] = {{.*}} ptr %{{[^ ]+}} device =
// CHECK:         mogg.tensor.store %[[PO]][%[[PI]]]
mo.graph @plain_store(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%c, %b) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}
