// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: kgen.kernel @buffer_size_dtype_folds
kgen.kernel @buffer_size_dtype_folds(%arg0: !meta.buffer<42, f32>, 
                              %arg1: !meta.buffer<?, f32>,
                              %arg2: !meta.buffer<42, ?>)
  -> (index, index, !kgen.dtype, !kgen.dtype) {
  // CHECK: %0 = meta.buffer.size %arg0 
  %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  // CHECK: %1 = meta.buffer.size %arg1 
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  
  // CHECK: %2 = meta.buffer.dtype %arg0
  %2 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  // CHECK: %3 = meta.buffer.dtype %arg2
  %3 = meta.buffer.dtype %arg2 : !meta.buffer<42, ?>

  // CHECK: kgen.return %0, %1, %2, %3
  kgen.return %0, %1, %2, %3 : index, index, !kgen.dtype, !kgen.dtype
}

// CHECK-LABEL: kgen.kernel @buffer_cast_folds
kgen.kernel @buffer_cast_folds(%arg0: !meta.buffer<?, ?>)
 -> (!meta.buffer<?, ?>, !meta.buffer<?, ?>) {
  // TODO: Should fold this away.
  // CHECK: %0 = meta.buffer.cast %arg0
  %0 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, ?>

  // TODO: Should fold this away.
  // CHECK: %1 = meta.buffer.cast %arg0
  %1 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  // CHECK: %2 = meta.buffer.cast %1
  %2 = meta.buffer.cast %1 : !meta.buffer<?, f32> to !meta.buffer<?, ?>

  // CHECK: kgen.return %0, %2
  kgen.return %0, %2 : !meta.buffer<?, ?>, !meta.buffer<?, ?>
}
