// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: kgen.func @buffer_size_dtype_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<{{.*}}>, %[[ARG1:.*]]: !zap.buffer<{{.*}}>, %[[ARG2:.*]]: !zap.buffer<{{.*}}>
kgen.func @buffer_size_dtype_folds(%arg0: !zap.buffer<42, f32>,
                              %arg1: !zap.buffer<?, f32>,
                              %arg2: !zap.buffer<42, ?>)
  -> (index, index, !kgen.dtype, !kgen.dtype) {
  // CHECK: %[[V0:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[V1:.*]] = kgen.param.constant = <42>

  %0 = zap.buffer.size %arg0 : !zap.buffer<42, f32>
  // CHECK: %[[V2:.*]] = zap.buffer.size %[[ARG1]]
  %1 = zap.buffer.size %arg1 : !zap.buffer<?, f32>

  %2 = zap.buffer.dtype %arg0 : !zap.buffer<42, f32>
  // CHECK: %[[V3:.*]] = zap.buffer.dtype %[[ARG2]]
  %3 = zap.buffer.dtype %arg2 : !zap.buffer<42, ?>

  // CHECK: kgen.return %[[V1]], %[[V2]], %[[V0]], %[[V3]]
  kgen.return %0, %1, %2, %3 : index, index, !kgen.dtype, !kgen.dtype
}

// CHECK-LABEL: kgen.func @buffer_bitcast_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<{{.*}}>, %[[ARG1:.*]]: !zap.buffer<{{.*}}
kgen.func @buffer_bitcast_folds(%arg0: !zap.buffer<?, ?>, %arg1: !zap.buffer<10, f32>)
 -> (!zap.buffer<?, ?>, !zap.buffer<?, ?>, !zap.buffer<?, ?>) {
  // Noop casts get folded away.
  %0 = zap.buffer.bitcast %arg0 : !zap.buffer<?, ?> to !zap.buffer<?, ?>

  // A-B-A cast.
  %1 = zap.buffer.bitcast %arg0 : !zap.buffer<?, ?> to !zap.buffer<?, f32>
  %2 = zap.buffer.bitcast %1 : !zap.buffer<?, f32> to !zap.buffer<?, ?>

  // A-B-C cast.
  // CHECK:  %[[V0:.*]] = zap.buffer.bitcast %[[ARG1]] : !zap.buffer<10, f32> to !zap.buffer<?, ?>
  %3 = zap.buffer.bitcast %arg1 : !zap.buffer<10, f32> to !zap.buffer<?, f32>
  %4 = zap.buffer.bitcast %3 : !zap.buffer<?, f32> to !zap.buffer<?, ?>

  // CHECK: kgen.return %[[ARG0]], %[[ARG0]], %[[V0]]
  kgen.return %0, %2, %4 : !zap.buffer<?, ?>, !zap.buffer<?, ?>, !zap.buffer<?, ?>
}

// CHECK-LABEL: kgen.func @tensor_dim_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.tensor<[42], f32>
// CHECK-SAME: %[[ARG1:.*]]: !zap.tensor<[42, ?], f32>
// CHECK-SAME: %[[ARG2:.*]]: !zap.tensor<[?, 42], f32>
kgen.func @tensor_dim_folds(%arg0: !zap.tensor<[42], f32>,
                            %arg1: !zap.tensor<[42, ?], f32>,
                            %arg2: !zap.tensor<[?, 42], f32>)
  -> (index, index, index, index, index) {
  // CHECK-DAG: %[[V0:.*]] = kgen.param.constant = <42>
  %0 = zap.tensor.dim %arg0[0] : !zap.tensor<[42], f32>
  %1 = zap.tensor.dim %arg1[0] : !zap.tensor<[42, ?], f32>
  // CHECK: %[[V3:.*]] = zap.tensor.dim %[[ARG1]][1]
  %2 = zap.tensor.dim %arg1[1] : !zap.tensor<[42, ?], f32>
  // CHECK: %[[V4:.*]] = zap.tensor.dim %[[ARG2]][0]
  %3 = zap.tensor.dim %arg2[0] : !zap.tensor<[?, 42], f32>
  %4 = zap.tensor.dim %arg2[1] : !zap.tensor<[?, 42], f32>

  // CHECK: kgen.return %[[V0]], %[[V0]], %[[V3]], %[[V4]], %[[V0]]
  kgen.return %0, %1, %2, %3, %4 : index, index, index, index, index
}


// CHECK-LABEL: kgen.func @tensor_dtype_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.tensor<[42], f32>
// CHECK-SAME: %[[ARG1:.*]]: !zap.tensor<[42, ?], f32>
// CHECK-SAME: %[[ARG2:.*]]: !zap.tensor<[?, 42], ?>
kgen.func @tensor_dtype_folds(%arg0: !zap.tensor<[42], f32>,
                              %arg1: !zap.tensor<[42, ?], f32>,
                              %arg2: !zap.tensor<[?, 42], ?>)
  -> (!kgen.dtype, !kgen.dtype, !kgen.dtype) {
  // CHECK-DAG: %[[V0:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK-DAG: %[[V1:.*]] = zap.tensor.dtype %[[ARG2]] : !zap.tensor<[?, 42], ?>

  %0 = zap.tensor.dtype %arg0 : !zap.tensor<[42], f32>
  %1 = zap.tensor.dtype %arg1 : !zap.tensor<[42, ?], f32>
  %2 = zap.tensor.dtype %arg2 : !zap.tensor<[?, 42], ?>

  // CHECK: kgen.return %[[V0]], %[[V0]], %[[V1]]
  kgen.return %0, %1, %2 : !kgen.dtype, !kgen.dtype, !kgen.dtype
}

// CHECK-LABEL: kgen.func @tensor_rank_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.tensor<[42], f32>
// CHECK-SAME: %[[ARG1:.*]]: !zap.tensor<[?, ?, ?], f32>
// CHECK-SAME: %[[ARG2:.*]]: !zap.tensor<[1, ?, 3, 4], ?>
kgen.func @tensor_rank_folds(%arg0: !zap.tensor<[42], f32>,
                             %arg1: !zap.tensor<[?, ?, ?], f32>,
                             %arg2: !zap.tensor<[1, ?, 3, 4], ?>)
  -> (index, index, index) {
  // CHECK-DAG: %[[V0:.*]] = kgen.param.constant = <1>
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant = <3>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant = <4>

  %0 = zap.tensor.rank %arg0 : !zap.tensor<[42], f32>
  %1 = zap.tensor.rank %arg1 : !zap.tensor<[?, ?, ?], f32>
  %2 = zap.tensor.rank %arg2 : !zap.tensor<[1, ?, 3, 4], ?>

  // CHECK: kgen.return %[[V0]], %[[V1]], %[[V2]]
  kgen.return %0, %1, %2 : index, index, index
}
