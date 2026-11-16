// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" -mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg --mogg-pipeline -mogg-to-mgp -canonicalize -mogg-outline-async-routines -mogg-add-try-except-blocks -mogg-emit-mojo -mogg-dump-mojo 2>&1 | FileCheck %s

// Test that a kernel with variadic inputs packs them into a TensorTuple in the
// emitted Mojo code.

// The kernel takes its three variadic int32 rank-1 inputs and its int32 rank-1
// output, each reconstructed via a typed tensor init, and the three inputs
// pack into one TensorTuple.
// CHECK: def {{__[A-Za-z0-9_]+}}(
// CHECK-COUNT-3: mogg_tensor_init[DType.int32, 1, {{.*}}extensibility.IO.Input, {{.*}}]
// CHECK: mogg_tensor_init[DType.int32, 1, {{.*}}extensibility.IO.Output, {{.*}}]
// CHECK: var [[INS:.*]] = extensibility.TensorTuple[type_of({{.*}}), type_of({{.*}}), type_of({{.*}})]({{.*}}, {{.*}}, {{.*}})
// CHECK: VariadicAdd.execute({{.*}}[[INS]]{{.*}})

mo.graph @variadic(%arg0: !mo.tensor<[4], si32>, %arg1: !mo.tensor<[4], si32>, %arg2: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %result = mo.custom [symbol = "variadic_add", device = #M.device_ref<"cpu", 0>](%arg0, %arg1, %arg2) : (
    !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>
  ) -> !mo.tensor<[4], si32>
  mo.output %result : !mo.tensor<[4], si32>
}
