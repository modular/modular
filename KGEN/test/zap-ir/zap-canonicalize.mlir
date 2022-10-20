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

// CHECK-LABEL: kgen.func @ndbuffer_dim_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.ndbuffer<[42], f32>
// CHECK-SAME: %[[ARG1:.*]]: !zap.ndbuffer<[42, ?], f32>
// CHECK-SAME: %[[ARG2:.*]]: !zap.ndbuffer<[?, 42], f32>
kgen.func @ndbuffer_dim_folds(%arg0: !zap.ndbuffer<[42], f32>,
                            %arg1: !zap.ndbuffer<[42, ?], f32>,
                            %arg2: !zap.ndbuffer<[?, 42], f32>)
  -> (index, index, index, index, index) {
  // CHECK-DAG: %[[V0:.*]] = kgen.param.constant = <42>
  %0 = zap.ndbuffer.dim %arg0[0] : !zap.ndbuffer<[42], f32>
  %1 = zap.ndbuffer.dim %arg1[0] : !zap.ndbuffer<[42, ?], f32>
  // CHECK: %[[V3:.*]] = zap.ndbuffer.dim %[[ARG1]][1]
  %2 = zap.ndbuffer.dim %arg1[1] : !zap.ndbuffer<[42, ?], f32>
  // CHECK: %[[V4:.*]] = zap.ndbuffer.dim %[[ARG2]][0]
  %3 = zap.ndbuffer.dim %arg2[0] : !zap.ndbuffer<[?, 42], f32>
  %4 = zap.ndbuffer.dim %arg2[1] : !zap.ndbuffer<[?, 42], f32>

  // CHECK: kgen.return %[[V0]], %[[V0]], %[[V3]], %[[V4]], %[[V0]]
  kgen.return %0, %1, %2, %3, %4 : index, index, index, index, index
}


// CHECK-LABEL: kgen.func @ndbuffer_dtype_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.ndbuffer<[42], f32>
// CHECK-SAME: %[[ARG1:.*]]: !zap.ndbuffer<[42, ?], f32>
// CHECK-SAME: %[[ARG2:.*]]: !zap.ndbuffer<[?, 42], ?>
kgen.func @ndbuffer_dtype_folds(%arg0: !zap.ndbuffer<[42], f32>,
                              %arg1: !zap.ndbuffer<[42, ?], f32>,
                              %arg2: !zap.ndbuffer<[?, 42], ?>)
  -> (!kgen.dtype, !kgen.dtype, !kgen.dtype) {
  // CHECK-DAG: %[[V0:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK-DAG: %[[V1:.*]] = zap.ndbuffer.dtype %[[ARG2]] : !zap.ndbuffer<[?, 42], ?>

  %0 = zap.ndbuffer.dtype %arg0 : !zap.ndbuffer<[42], f32>
  %1 = zap.ndbuffer.dtype %arg1 : !zap.ndbuffer<[42, ?], f32>
  %2 = zap.ndbuffer.dtype %arg2 : !zap.ndbuffer<[?, 42], ?>

  // CHECK: kgen.return %[[V0]], %[[V0]], %[[V1]]
  kgen.return %0, %1, %2 : !kgen.dtype, !kgen.dtype, !kgen.dtype
}

// CHECK-LABEL: kgen.func @ndbuffer_rank_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.ndbuffer<[42], f32>
// CHECK-SAME: %[[ARG1:.*]]: !zap.ndbuffer<[?, ?, ?], f32>
// CHECK-SAME: %[[ARG2:.*]]: !zap.ndbuffer<[1, ?, 3, 4], ?>
kgen.func @ndbuffer_rank_folds(%arg0: !zap.ndbuffer<[42], f32>,
                             %arg1: !zap.ndbuffer<[?, ?, ?], f32>,
                             %arg2: !zap.ndbuffer<[1, ?, 3, 4], ?>)
  -> (index, index, index) {
  // CHECK-DAG: %[[V0:.*]] = kgen.param.constant = <1>
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant = <3>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant = <4>

  %0 = zap.ndbuffer.rank %arg0 : !zap.ndbuffer<[42], f32>
  %1 = zap.ndbuffer.rank %arg1 : !zap.ndbuffer<[?, ?, ?], f32>
  %2 = zap.ndbuffer.rank %arg2 : !zap.ndbuffer<[1, ?, 3, 4], ?>

  // CHECK: kgen.return %[[V0]], %[[V1]], %[[V2]]
  kgen.return %0, %1, %2 : index, index, index
}


// CHECK-LABEL: kgen.func @ndbuffer_bitcast_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.ndbuffer<{{.*}}>, %[[ARG1:.*]]: !zap.ndbuffer<{{.*}}
kgen.func @ndbuffer_bitcast_folds(%arg0: !zap.ndbuffer<[?, ?], f32>, %arg1: !zap.ndbuffer<[10, 42], f32>)
 -> (!zap.ndbuffer<[?, ?], f32>, !zap.ndbuffer<[?, ?], f32>, !zap.ndbuffer<[?, ?], si64>) {
  // Noop casts get folded away.
  %0 = zap.ndbuffer.bitcast %arg0 : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[?, ?], f32>

  // A-B-A cast.
  %1 = zap.ndbuffer.bitcast %arg0 : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[?, ?], si64>
  %2 = zap.ndbuffer.bitcast %1 : !zap.ndbuffer<[?, ?], si64> to !zap.ndbuffer<[?, ?], f32>

  // A-B-C cast.
  // CHECK:  %[[V0:.*]] = zap.ndbuffer.bitcast %[[ARG1]] : !zap.ndbuffer<[10, 42], f32> to !zap.ndbuffer<[?, ?], si64>
  %3 = zap.ndbuffer.bitcast %arg1 : !zap.ndbuffer<[10, 42], f32> to !zap.ndbuffer<[?, ?], f32>
  %4 = zap.ndbuffer.bitcast %3 : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[?, ?], si64>

  // CHECK: kgen.return %[[ARG0]], %[[ARG0]], %[[V0]]
  kgen.return %0, %2, %4 : !zap.ndbuffer<[?, ?], f32>, !zap.ndbuffer<[?, ?], f32>, !zap.ndbuffer<[?, ?], si64>
}
