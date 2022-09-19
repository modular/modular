// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: kgen.func @buffer_size_dtype_folds
// CHECK-SAME: %[[ARG0:.*]]: !meta.buffer<{{.*}}>, %[[ARG1:.*]]: !meta.buffer<{{.*}}>, %[[ARG2:.*]]: !meta.buffer<{{.*}}>
kgen.func @buffer_size_dtype_folds(%arg0: !meta.buffer<42, f32>,
                              %arg1: !meta.buffer<?, f32>,
                              %arg2: !meta.buffer<42, ?>)
  -> (index, index, !kgen.dtype, !kgen.dtype) {
  // CHECK: %[[V0:.*]] = kgen.param.constant : dtype = <f32>
  // CHECK: %[[V1:.*]] = kgen.param.constant = <42>

  %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  // CHECK: %[[V2:.*]] = meta.buffer.size %[[ARG1]]
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>

  %2 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  // CHECK: %[[V3:.*]] = meta.buffer.dtype %[[ARG2]]
  %3 = meta.buffer.dtype %arg2 : !meta.buffer<42, ?>

  // CHECK: kgen.return %[[V1]], %[[V2]], %[[V0]], %[[V3]]
  kgen.return %0, %1, %2, %3 : index, index, !kgen.dtype, !kgen.dtype
}

// CHECK-LABEL: @rebind_folds
kgen.generator @rebind_folds<dtype: dtype, type: type>(
  %a: i32, %b: !meta.scalar<f32>, %c: !meta.scalar<dtype>, %d: !kgen.paramref<type>
) -> (
  i32, !meta.scalar<f32>, !meta.scalar<dtype>, !kgen.paramref<type>
) {
  // CHECK-NOT: kgen.rebind
  %0 = kgen.rebind %a : i32 to i32
  %1 = kgen.rebind %b : !meta.scalar<f32> to !meta.scalar<f32>
  %2 = kgen.rebind %c : !meta.scalar<dtype> to !meta.scalar<dtype>
  %3 = kgen.rebind %d : !kgen.paramref<type> to !kgen.paramref<type>
  kgen.return %0, %1, %2, %3 : i32, !meta.scalar<f32>, !meta.scalar<dtype>, !kgen.paramref<type>
}

// CHECK-LABEL: kgen.func @buffer_rebind_folds
// CHECK-SAME: %[[ARG0:.*]]: !meta.buffer<{{.*}}>, %[[ARG1:.*]]: !meta.buffer<{{.*}}
kgen.func @buffer_rebind_folds(%arg0: !meta.buffer<?, ?>, %arg1: !meta.buffer<10, f32>)
 -> (!meta.buffer<?, ?>, !meta.buffer<?, ?>, !meta.buffer<?, ?>) {
  // Noop casts get folded away.
  %0 = meta.buffer.convert %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, ?>

  // A-B-A cast.
  %1 = meta.buffer.convert %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  %2 = meta.buffer.convert %1 : !meta.buffer<?, f32> to !meta.buffer<?, ?>

  // A-B-C cast.
  // CHECK:  %[[V0:.*]] = meta.buffer.convert %[[ARG1]] : !meta.buffer<10, f32> to !meta.buffer<?, ?>
  %3 = meta.buffer.convert %arg1 : !meta.buffer<10, f32> to !meta.buffer<?, f32>
  %4 = meta.buffer.convert %3 : !meta.buffer<?, f32> to !meta.buffer<?, ?>

  // CHECK: kgen.return %[[ARG0]], %[[ARG0]], %[[V0]]
  kgen.return %0, %2, %4 : !meta.buffer<?, ?>, !meta.buffer<?, ?>, !meta.buffer<?, ?>
}


// CHECK-LABEL: kgen.func @meta_cast_from_folds
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @meta_cast_from_folds(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {

  // A-B-A cast.
  %1 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<f32>

  // TODO: Update return check once meta.scalar.cast is implemented.
  // CHECK: kgen.return %[[ARG0]]
  kgen.return %2 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @meta_cast_to_folds
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32 {
kgen.func @meta_cast_to_folds(%arg0: f32) -> f32 {

  // A-B-A cast.
  %1 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f32> to f32

  // CHECK: kgen.return %[[ARG0]]
  kgen.return %2 : f32
}

kgen.func @producesResultParam<() -> index>() {
  kgen.return<result = 42>
}


// CHECK-LABEL: kgen.generator @param_assert_simplify<p1: i1, p2>()
// CHECK-NEXT: constraints <
// CHECK-NEXT:   [p1, "this is a constraint!", #
// CHECK-NEXT:   [eq(add(p2, 4), 17), "also a constraint", #
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

kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @call_param_canonicalize
kgen.generator @call_param_canonicalize(%arg0: si32) -> si32 {
  // CHECK: %0 = kgen.call @trivial(%arg0) : (si32) -> si32
  %0 = kgen.call_param[(si32) -> si32: @trivial](%arg0)
  kgen.return %0: si32
}
