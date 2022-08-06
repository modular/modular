// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: kgen.kernel @buffer_size_dtype_folds
kgen.kernel @buffer_size_dtype_folds(%arg0: !meta.buffer<42, f32>,
                              %arg1: !meta.buffer<?, f32>,
                              %arg2: !meta.buffer<42, ?>)
  -> (index, index, !kgen.dtype, !kgen.dtype) {
  // CHECK: %0 = kgen.param.value : dtype = <f32>
  // CHECK: %1 = kgen.param.value = <42>

  %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  // CHECK: %2 = meta.buffer.size %arg1
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>

  %2 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  // CHECK: %3 = meta.buffer.dtype %arg2
  %3 = meta.buffer.dtype %arg2 : !meta.buffer<42, ?>

  // CHECK: kgen.return %1, %2, %0, %3
  kgen.return %0, %1, %2, %3 : index, index, !kgen.dtype, !kgen.dtype
}

// CHECK-LABEL: kgen.kernel @buffer_cast_folds
kgen.kernel @buffer_cast_folds(%arg0: !meta.buffer<?, ?>, %arg1: !meta.buffer<10, f32>)
 -> (!meta.buffer<?, ?>, !meta.buffer<?, ?>, !meta.buffer<?, ?>) {
  // Noop casts get folded away.
  %0 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, ?>

  // A-B-A cast.
  %1 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  %2 = meta.buffer.cast %1 : !meta.buffer<?, f32> to !meta.buffer<?, ?>

  // A-B-C cast.
  // CHECK:  %0 = meta.buffer.cast %arg1 : !meta.buffer<10, f32> to !meta.buffer<?, ?>
  %3 = meta.buffer.cast %arg1 : !meta.buffer<10, f32> to !meta.buffer<?, f32>
  %4 = meta.buffer.cast %3 : !meta.buffer<?, f32> to !meta.buffer<?, ?>

  // CHECK: kgen.return %arg0, %arg0, %0
  kgen.return %0, %2, %4 : !meta.buffer<?, ?>, !meta.buffer<?, ?>, !meta.buffer<?, ?>
}


// CHECK-LABEL: kgen.kernel @meta_cast_from_folds(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @meta_cast_from_folds(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {

  // A-B-A cast.
  %1 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<f32>

  // TODO: Update return check once meta.scalar.cast is implemented.
  // CHECK: kgen.return %arg0
  kgen.return %2 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @meta_cast_to_folds(%arg0: f32) -> f32 {
kgen.kernel @meta_cast_to_folds(%arg0: f32) -> f32 {

  // A-B-A cast.
  %1 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f32> to f32

  // CHECK: kgen.return %arg0
  kgen.return %2 : f32
}

kgen.kernel @producesResultParam<() -> result>() {
  kgen.return<result = 42>
}


// CHECK-LABEL: kgen.generator @param_assert_simplify<p1: i1, p2>()
// CHECK-NEXT: constraints <
// CHECK-NEXT:   p1, "this is a constraint!",
// CHECK-NEXT:   eq(add(p2, 4), 17), "also a constraint"> { 
kgen.generator @param_assert_simplify<p1 : i1, p2>() {

  kgen.param.assert <p1>, "this is a constraint!"
  kgen.param.assert <eq(add(p2, 4), 17)>, "also a constraint"

  kgen.param.assert <1>, "this is pointless"

  // CHECK-NEXT:   kgen.param.assert <0>, "failing asserts must be kept"
  kgen.param.assert <eq(42, 41)>, "failing asserts must be kept"

  // CHECK-NEXT: kgen.call @producesResultParam
  kgen.call @producesResultParam<() -> result>() : () -> ()

  // CHECK-NEXT: kgen.param.assert <eq(result, 12)>, "this stays"
  kgen.param.assert <eq(result, 12)>, "this stays"
  kgen.return
}
