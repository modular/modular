// RUN: mo-opt %s --mogg-import-packages -mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg --mogg-pipeline -mogg-to-mgp -canonicalize -mogg-outline-async-routines -mogg-add-try-except-blocks -mogg-emit-mojo -mogg-dump-mojo 2>&1 | FileCheck %s

// Test that mo.concat with variadic inputs properly generates VariadicTensors in the emitted Mojo code

// The concat kernel takes its three variadic int32 rank-1 inputs and its
// int32 rank-1 output, each reconstructed via a typed tensor init.
// CHECK: def {{__[A-Za-z0-9_]+}}(
// CHECK-COUNT-3: mogg_tensor_init[DType.int32, 1, {{.*}}extensibility.IO.Input, {{.*}}]
// CHECK: mogg_tensor_init[DType.int32, 1, {{.*}}extensibility.IO.Output, {{.*}}]
// CHECK: var {{.*}} = std.utils.StaticTuple[size = 3]({{.*}}._ptr, {{.*}}._ptr, {{.*}}._ptr)
// CHECK: var {{.*}} = std.utils.StaticTuple[size = 3]({{.*}}.shape(), {{.*}}.shape(), {{.*}}.shape())
// CHECK: comptime {{.*}} = ParameterList.of[{{.*}}.static_spec.get_internals(), {{.*}}.static_spec.get_internals(), {{.*}}.static_spec.get_internals()]()
// CHECK: var {{.*}} = managed_tensor_slice._FusedInputVariadicTensors[{{.*}}, static_specs = {{.*}}]({{.*}}, {{.*}}, {{.*}})
// CHECK: builtin_kernels.gather_scatter.Concat.execute[target = "cpu", axis = 0]({{.*}}, {{.*}}, {{.*}})

mo.graph @concat(%arg0: !mo.tensor<[2], si32>, %arg1: !mo.tensor<[3], si32>, %arg2: !mo.tensor<[4], si32>) -> !mo.tensor<[9], si32> {
  %axis = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %result = mo.concat[0](%arg0, %arg1, %arg2) : (
    !mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[4], si32>
  ) -> !mo.tensor<[9], si32>
  mo.output %result : !mo.tensor<[9], si32>
}
