// RUN: mo-opt %s --split-input-file --mo-shape-inference --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// A constant perm is fully static, so the transpose view is built from its
// input alone.

// CHECK: var [[VIEW:[a-z_0-9]+]] = {{.*}}mogg_tensor_create_transpose[output_static_shape = {{[a-z_0-9]+}}, static_permutations = {{[a-z_0-9]+}}]({{[a-z_0-9]+}}){{$}}
mo.graph @transpose_static(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// -----

// A graph-argument perm is read at run time, so the view also takes it.

// CHECK: var [[VIEW:[a-z_0-9]+]] = {{.*}}mogg_tensor_create_transpose[output_static_shape = {{[a-z_0-9]+}}, static_permutations = {{[a-z_0-9]+}}]({{[a-z_0-9]+}}, {{[a-z_0-9]+}}){{$}}
mo.graph @transpose_dynamic<D0, D1 -> D2, D3>(
    %input: !mo.tensor<[D0, D1], f32>,
    %perm: !mo.tensor<[2], si64>)
    -> !mo.tensor<[D2, D3], f32> {
  %res = mo.transpose<() -> D4, D5>(%input, %perm) : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
