// RUN: kgen-opt %s -split-input-file -prune-impossible-variants | FileCheck %s

// CHECK-LABEL: kgen.func export @known_true_or_false
kgen.func export @known_true_or_false(%arg0: i32, %arg1: f32, %arg2: i8) -> i1 {
  %0 = kgen.variant.create %arg0, 1 : <i8, i32>
  // CHECK: %[[TRUE:.*]] = kgen.param.constant: i1 = <1>
  %1 = kgen.variant.is %0, 1 : <i8, i32>
  // CHECK: hlcf.if %[[TRUE]]
  %2 = hlcf.if %1 -> !kgen.variant<f32, i8> {
    %3 = kgen.variant.create %arg2, 1 : <f32, i8>
    hlcf.yield %3 : !kgen.variant<f32, i8>
  } else {
    %3 = kgen.variant.create %arg1, 0 : <f32, i8>
    hlcf.yield %3 : !kgen.variant<f32, i8>
  }
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: i1 = <0>
  %5 = kgen.variant.is %2, 0 : <f32, i8>
  // CHECK: return %[[FALSE]]
  kgen.return %5 : i1
}

// -----

// CHECK-LABEL: kgen.func export @known_false
kgen.func export @known_false(%arg0: i32, %arg1: i8, %arg2: i1) -> i1 {
  %0 = hlcf.if %arg2 -> !kgen.variant<i8, i32, f32> {
    %1 = kgen.variant.create %arg0, 1 : <i8, i32, f32>
    hlcf.yield %1 : !kgen.variant<i8, i32, f32>
  } else {
    %1 = kgen.variant.create %arg1, 0 : <i8, i32, f32>
    hlcf.yield %1 : !kgen.variant<i8, i32, f32>
  }
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: i1 = <0>
  %2 = kgen.variant.is %0, 2 : <i8, i32, f32>
  // CHECK: return %[[FALSE]]
  kgen.return %2 : i1
}

// -----

// CHECK-LABEL: kgen.func @always_index
// CHECK-SAME: -> index
kgen.func @always_index() -> !kgen.variant<i8, index> {
  %0 = index.constant 5
  %1 = kgen.variant.create %0, 1 : <i8, index>
  // CHECK: return %{{.*}} : index
  kgen.return %1 : !kgen.variant<i8, index>
}

// CHECK-LABEL: kgen.func @entry
// CHECK-SAME: -> i32
kgen.func @entry() -> !kgen.variant<i8, i32> {
  // CHECK: %[[RESULT:.*]] = kgen.call
  // CHECK-SAME: -> index
  %0 = kgen.call @always_index() : () -> !kgen.variant<i8, index>
  // CHECK: kgen.variant.create %[[RESULT]]
  %1 = kgen.variant.is %0, 0 : <i8, index>
  // CHECK: %[[IF_RESULT:.*]] = hlcf.if
  %2 = hlcf.if %1 -> !kgen.variant<i8, i32> {
    %3 = kgen.variant.take %0, 0 : <i8, index>
    %4 = kgen.variant.create %3, 0 : <i8, i32>
    hlcf.yield %4 : !kgen.variant<i8, i32>
  } else {
    %3 = kgen.param.constant: i32 = <0>
    %5 = kgen.variant.create %3, 1 : <i8, i32>
    hlcf.yield %5 : !kgen.variant<i8, i32>
  }
  // CHECK: %[[RETURN:.*]] = kgen.variant.take %[[IF_RESULT]], 1 : <i8, i32>
  // CHECK: return %[[RETURN]]
  kgen.return %2 : !kgen.variant<i8, i32>
}

// CHECK-LABEL: kgen.func export @public
kgen.func export @public() {
  // CHECK-NEXT: @entry() : () -> i32
  %0 = kgen.call @entry() : () -> !kgen.variant<i8, i32>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @do_not_rewrite
// CHECK-SAME: -> !kgen.variant<index>
kgen.func @do_not_rewrite() -> !kgen.variant<index> {
  %0 = index.constant 0
  %1 = kgen.variant.create %0, 0 : <index>
  kgen.return %1 : !kgen.variant<index>
}

// CHECK-LABEL: kgen.func export @call
kgen.func export @call() {
  // CHECK-NEXT: constant: () -> !kgen.variant<index> = <@do_not_rewrite>
  %0 = kgen.param.constant: () -> !kgen.variant<index> = <@do_not_rewrite>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @dead_code
// CHECK-SAME: -> !kgen.variant<index>
kgen.func @dead_code() -> !kgen.variant<index> {
  %0 = index.constant 0
  %1 = kgen.variant.create %0, 0 : <index>
  // CHECK: return %{{.*}} : !kgen.variant<index>
  kgen.return %1 : !kgen.variant<index>
}

// -----

kgen.func @always_i32(%a: i32) -> !kgen.variant<i32, f32> {
  %0 = kgen.variant.create %a, 0 : <i32, f32>
  kgen.return %0 : !kgen.variant<i32, f32>
}

// Make sure all callsites are rewritten.

// CHECK-LABEL: kgen.func export @first_callsite
kgen.func export @first_callsite(%a: i32) {
  // CHECK: @always_i32(%arg0) : (i32) -> i32
  %0 = kgen.call @always_i32(%a) : (i32) -> !kgen.variant<i32, f32>
  kgen.return
}

// CHECK-LABEL: kgen.func export @second_callsite
kgen.func export @second_callsite(%a: i32) {
  // CHECK: @always_i32(%arg0) : (i32) -> i32
  %0 = kgen.call @always_i32(%a) : (i32) -> !kgen.variant<i32, f32>
  %1 = kgen.call @always_i32(%a) : (i32) -> !kgen.variant<i32, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @multiple_returns
// CHECK-SAME: -> i32
kgen.func @multiple_returns(%a: i32, %b: i64) -> !kgen.variant<i32, i64> {
  %0 = kgen.variant.create %a, 0 : <i32, i64>
  %1 = hlcf.loop (%arg0 = %0 : !kgen.variant<i32, i64>) -> !kgen.variant<i32, i64> {
    %2 = kgen.variant.is %arg0, 0 : <i32, i64>
    // CHECK: hlcf.if
    hlcf.if %2 {
      // CHECK-NEXT: %[[RES:.*]] = kgen.variant.take %arg2, 0 : <i32, i64>
      // CHECK-NEXT: kgen.return %[[RES]]
      kgen.return %arg0 : !kgen.variant<i32, i64>
    } else {
      hlcf.yield
    }
    // COM: This is valid because the code is dead.
    // CHECK: %[[I64:.*]] = kgen.variant.create %arg1, 1 : <i32, i64>
    // CHECK-NEXT: %[[I32:.*]] = kgen.variant.take %[[I64]], 0 : <i32, i64>
    // CHECK-NEXT: kgen.return %[[I32]]
    %3 = kgen.variant.create %b, 1 : <i32, i64>
    kgen.return %3 : !kgen.variant<i32, i64>
  }
  // CHECK: %[[RES:.*]] = kgen.variant.take %1, 0 : <i32, i64>
  // CHECK-NEXT: kgen.return %[[RES]]
  kgen.return %1 : !kgen.variant<i32, i64>
}

// CHECK-LABEL: kgen.func export @call_it
kgen.func export @call_it(%a: i32, %b: i64) {
  // CHECK: kgen.call @multiple_returns
  // CHECK-SAME: (i32, i64) -> i32
  %0 = kgen.call @multiple_returns(%a, %b) : (i32, i64) -> !kgen.variant<i32, i64>
  kgen.return
}
