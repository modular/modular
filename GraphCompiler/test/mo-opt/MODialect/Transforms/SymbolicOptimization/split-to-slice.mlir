// RUN: mo-opt -split-input-file --mo-symbolic-optimize %s | FileCheck %s

mo.graph @split_to_slice<N>(%input: !mo.tensor<[N], si32>) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>) {
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = -1] : (
    !mo.tensor<[N], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>)

  %o1 = mo.relu(%out1): !mo.tensor<[1], si32>
  %o2 = mo.relu(%out2): !mo.tensor<[2], si32>
  %o3 = mo.relu(%out3): !mo.tensor<[3], si32>
  mo.output %o1, %o2, %o3 : !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
}

// CHECK-LABEL: mo.graph @split_to_slice<N>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[N], si32
// CHECK-DAG: %[[ONE:.*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-DAG: %[[ZERO:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-DAG: %[[THREE:.*]] = mo.constant <#M.dense_array<3> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-DAG: %[[SIX:.*]] = mo.constant <#M.dense_array<6> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK: %[[O0:.*]] = mo.slice(%[[INPUT]], %[[ZERO]], %[[ONE]], %[[ONE]]) : (!mo.tensor<[N], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: %[[O1:.*]] = mo.slice(%[[INPUT]], %[[ONE]], %[[THREE]], %[[ONE]]) : (!mo.tensor<[N], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si32>
// CHECK-NEXT: %[[O2:.*]] = mo.slice(%[[INPUT]], %[[THREE]], %[[SIX]], %[[ONE]]) : (!mo.tensor<[N], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si32>

// -----

mo.graph @split_to_slice_with_device<N>(%input: !mo.tensor<[N], si32, dev:0>) -> (!mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>) {
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = -1] : (
    !mo.tensor<[N], si32, dev:0>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>)

  %o1 = mo.relu(%out1) : !mo.tensor<[1], si32, dev:0>
  %o2 = mo.relu(%out2) : !mo.tensor<[2], si32, dev:0>
  %o3 = mo.relu(%out3) : !mo.tensor<[3], si32, dev:0>
  mo.output %o1, %o2, %o3 : !mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>
}

// CHECK-LABEL: mo.graph @split_to_slice_with_device<N>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[N], si32, dev:0>
// CHECK-DAG: %[[ONE:.*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>>{{.*}} : !mo.tensor<[1], si64>
// CHECK-DAG: %[[ZERO:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi64>>{{.*}} : !mo.tensor<[1], si64>
// CHECK-DAG: %[[THREE:.*]] = mo.constant <#M.dense_array<3> : !M.array<1xsi64>>{{.*}} : !mo.tensor<[1], si64>
// CHECK-DAG: %[[SIX:.*]] = mo.constant <#M.dense_array<6> : !M.array<1xsi64>>{{.*}} : !mo.tensor<[1], si64>
// CHECK-NEXT: %[[O0:.*]] = mo.slice(%[[INPUT]], %[[ZERO]], %[[ONE]], %[[ONE]]) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si32, dev:0>
// CHECK-NEXT: %[[O1:.*]] = mo.slice(%[[INPUT]], %[[ONE]], %[[THREE]], %[[ONE]]) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si32, dev:0>
// CHECK-NEXT: %[[O2:.*]] = mo.slice(%[[INPUT]], %[[THREE]], %[[SIX]], %[[ONE]]) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si32, dev:0>

// -----

mo.graph @split_to_slice_3D<N, H, K>(%input: !mo.tensor<[N, H, K], si32>) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>) {
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = -1] : (
    !mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>)

  %o1 = mo.relu(%out1): !mo.tensor<[N, H, 1], si32>
  %o2 = mo.relu(%out2): !mo.tensor<[N, H, 2], si32>
  %o3 = mo.relu(%out3): !mo.tensor<[N, H, 3], si32>
  mo.output %o1, %o2, %o3 : !mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>
}
// CHECK-LABEL: mo.graph @split_to_slice_3D<N, H, K>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[N, H, K], si32
// CHECK-DAG: %[[STRIDE:.*]] = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[START0:.*]] = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[STOP0:.*]] = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[START1:.*]] = mo.constant <#M.dense_array<0, 0, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[STOP1:.*]] = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 3> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[START2:.*]] = mo.constant <#M.dense_array<0, 0, 3> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[STOP2:.*]] = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 6> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK: mo.slice(%[[INPUT]], %[[START0]], %[[STOP0]], %[[STRIDE]]) : (!mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, H, 1], si32>
// CHECK-NEXT: mo.slice(%[[INPUT]], %[[START1]], %[[STOP1]], %[[STRIDE]]) : (!mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, H, 2], si32>
// CHECK-NEXT: mo.slice(%[[INPUT]], %[[START2]], %[[STOP2]], %[[STRIDE]]) : (!mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, H, 3], si32>

// -----

mo.graph @split_to_slice_3D_axis1<N, H, K>(%input: !mo.tensor<[N, H, K], si32>) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>) {
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = 1] : (
    !mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>)

  %o1 = mo.relu(%out1): !mo.tensor<[N, H, 1], si32>
  %o2 = mo.relu(%out2): !mo.tensor<[N, H, 2], si32>
  %o3 = mo.relu(%out3): !mo.tensor<[N, H, 3], si32>
  mo.output %o1, %o2, %o3 : !mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>
}
// CHECK-LABEL: mo.graph @split_to_slice_3D_axis1<N, H, K>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[N, H, K], si32
// CHECK-DAG: %[[STRIDE:.*]] = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[START0:.*]] = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[STOP0:.*]] = mo.constant <#M.dense_array<9223372036854775807, 1, 9223372036854775807> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[START1:.*]] = mo.constant <#M.dense_array<0, 1, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[STOP1:.*]] = mo.constant <#M.dense_array<9223372036854775807, 3, 9223372036854775807> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[START2:.*]] = mo.constant <#M.dense_array<0, 3, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG: %[[STOP2:.*]] = mo.constant <#M.dense_array<9223372036854775807, 6, 9223372036854775807> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
// CHECK: mo.slice(%[[INPUT]], %[[START0]], %[[STOP0]], %[[STRIDE]]) : (!mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, H, 1], si32>
// CHECK-NEXT: mo.slice(%[[INPUT]], %[[START1]], %[[STOP1]], %[[STRIDE]]) : (!mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, H, 2], si32>
// CHECK-NEXT: mo.slice(%[[INPUT]], %[[START2]], %[[STOP2]], %[[STRIDE]]) : (!mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, H, 3], si32>

// -----

// Check we don't run when the axis is 0, handled by the MGP optimization.
mo.graph @no_run_split_to_slice_3D_axis0<N, H, K>(%input: !mo.tensor<[N, H, K], si32>) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>) {
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = 0] : (
    !mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>)

  %o1 = mo.relu(%out1): !mo.tensor<[N, H, 1], si32>
  %o2 = mo.relu(%out2): !mo.tensor<[N, H, 2], si32>
  %o3 = mo.relu(%out3): !mo.tensor<[N, H, 3], si32>
  mo.output %o1, %o2, %o3 : !mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>
}
// CHECK-LABEL: mo.graph @no_run_split_to_slice_3D_axis0<N, H, K>
// CHECK: mo.split


// Check if it is only used in the output we don't convert to slices.
mo.graph @no_run_split_to_slice_used_in_out<N, H, K>(%input: !mo.tensor<[N, H, K], si32>) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>) {
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = -1] : (
    !mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>)

  mo.output %out1, %out2, %out3 : !mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>
}
// CHECK-LABEL: mo.graph @no_run_split_to_slice_used_in_out<N, H, K>
// CHECK: mo.split

mo.graph @no_run_split_to_slice_dyn_splits<N, H, K>(%input: !mo.tensor<[N, H, K], si32>, %splitSizes: !mo.tensor<[3], si64>) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>) {
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = -1] : (
    !mo.tensor<[N, H, K], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>)
  %o1 = mo.relu(%out1): !mo.tensor<[N, H, 1], si32>
  %o2 = mo.relu(%out2): !mo.tensor<[N, H, 2], si32>
  %o3 = mo.relu(%out3): !mo.tensor<[N, H, 3], si32>
  mo.output %o1, %o2, %o3 : !mo.tensor<[N, H, 1], si32>, !mo.tensor<[N, H, 2], si32>, !mo.tensor<[N, H, 3], si32>
}
// CHECK-LABEL: mo.graph @no_run_split_to_slice_dyn_splits<N, H, K>
// CHECK: mo.split
