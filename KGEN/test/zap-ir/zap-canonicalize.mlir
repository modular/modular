// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: kgen.func @buffer_size_dtype_folds
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<{{.*}}>, %[[ARG1:.*]]: !zap.buffer<{{.*}}>, %[[ARG2:.*]]: !zap.buffer<{{.*}}>
kgen.func @buffer_size_dtype_folds(%arg0: !zap.buffer<42, f32>,
                              %arg1: !zap.buffer<?, f32>,
                              %arg2: !zap.buffer<42, ?>)
  -> (index, index, !kgen.dtype, !kgen.dtype) {
  // CHECK: %[[V0:.*]] = kgen.param.constant : dtype = <f32>
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
