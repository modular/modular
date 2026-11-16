// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%view_kernel_pkg" --mo-to-map | FileCheck %s

// A view op with a registered kernel is called like any other opaque kernel,
// instead of being lowered to its index loop. %view_kernel_pkg registers
// mo.static.broadcast_to and mo.transpose for GPU only, so on gpu:0 the ops
// become map.iter.opaque, and on the host, where no registration matches, the
// same ops still lower to map.index.broadcast / map.index.transpose.

// CHECK-LABEL: mo.graph @broadcastWithKernel
// CHECK-SAME:  (%[[IN:.*]]: !mo.tensor<[1, 64], f32, gpu:0>)
// CHECK:       map.iter.opaque
// CHECK:       inputs (%[[INARG:.*]]: !mo.tensor<[1, 64], f32, gpu:0>)
// CHECK:       outputs (%[[OUTARG:.*]] : !mo.tensor<[8, 64], f32, gpu:0>) {
// CHECK:       map.call.opaque @{{.*}}StaticBroadcastTo[] inputs(%[[INARG]]) outputs(%[[OUTARG]]) kernelName = "mo.static.broadcast_to"
// CHECK-NOT:   map.index.broadcast
mo.graph @broadcastWithKernel(%input: !mo.tensor<[1, 64], f32, gpu:0>) -> !mo.tensor<[8, 64], f32, gpu:0> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 64], f32, gpu:0> -> !mo.tensor<[8, 64], f32, gpu:0>
  mo.output %res : !mo.tensor<[8, 64], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @transposeWithKernel
// CHECK:       map.iter.opaque
// CHECK:       map.call.opaque @{{.*}}Transpose[static_permutations = [1, 0]] inputs(%{{.*}}, %{{.*}}) outputs(%{{.*}}) kernelName = "mo.transpose"
// CHECK-NOT:   map.index.transpose
mo.graph @transposeWithKernel(%input: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[3, 2], f32, gpu:0> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32, gpu:0>
  mo.output %res : !mo.tensor<[3, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @broadcastWithoutKernel
// CHECK-NOT:   map.iter.opaque
// CHECK:       map.iter.elementwise<[8, 64]>
// CHECK:       map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>>
// CHECK-NOT:   map.iter.opaque
mo.graph @broadcastWithoutKernel(%input: !mo.tensor<[1, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}

// CHECK-LABEL: mo.graph @transposeWithoutKernel
// CHECK-NOT:   map.iter.opaque
// CHECK:       map.iter.elementwise<[3, 2]>
// CHECK:       map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>>
// CHECK-NOT:   map.iter.opaque
mo.graph @transposeWithoutKernel(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}
