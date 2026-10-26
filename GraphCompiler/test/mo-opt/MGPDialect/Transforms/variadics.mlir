// RUN: mo-opt %s --mogg-import-packages -mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg --mogg-pipeline -mogg-to-mgp -canonicalize -mogg-outline-async-routines -mogg-add-try-except-blocks -mogg-emit-mojo -mogg-dump-mojo 2>&1 | FileCheck %s

// Test that mo.concat with variadic inputs properly generates a TensorTuple in the emitted Mojo code

// The concat kernel takes its three variadic int32 rank-1 inputs and its
// int32 rank-1 output, each reconstructed via a typed tensor init. Each input
// is a fused load wrapper, and the three wrappers pack into one TensorTuple.
// CHECK: def {{__[A-Za-z0-9_]+}}(
// CHECK-COUNT-3: mogg_tensor_init[DType.int32, 1, {{.*}}extensibility.IO.Input, {{.*}}]
// CHECK: mogg_tensor_init[DType.int32, 1, {{.*}}extensibility.IO.Output, {{.*}}]
// CHECK-COUNT-3: var {{.*}} = _MoggInputLoadFusionTensor[DType.int32, {{.*}}]
// CHECK: var [[OUT:.*]] = _MoggOutputStoreFusionTensor[DType.int32, {{.*}}]
// CHECK: var [[INS:.*]] = extensibility.TensorTuple[type_of({{.*}}), type_of({{.*}}), type_of({{.*}})]({{.*}}, {{.*}}, {{.*}})
// CHECK: builtin_kernels.gather_scatter.Concat.execute[target = "cpu", axis = 0]([[OUT]], [[INS]], {{.*}})

mo.graph @concat(%arg0: !mo.tensor<[2], si32>, %arg1: !mo.tensor<[3], si32>, %arg2: !mo.tensor<[4], si32>) -> !mo.tensor<[9], si32> {
  %axis = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %result = mo.concat[0](%arg0, %arg1, %arg2) : (
    !mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[4], si32>
  ) -> !mo.tensor<[9], si32>
  mo.output %result : !mo.tensor<[9], si32>
}
