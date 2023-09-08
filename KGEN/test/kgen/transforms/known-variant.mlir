// RUN: kgen-opt %s -split-input-file -prune-impossible-variants | FileCheck %s

// CHECK-LABEL: kgen.func export @known_true_or_false
kgen.func export @known_true_or_false(%arg0: i32, %arg1: f32, %arg2: i8) -> i1 {
  %0 = pop.variant.create %arg0 : i32 -> !pop.variant<i8, i32>
  // CHECK: %[[TRUE:.*]] = kgen.param.constant: i1 = <1>
  %1 = pop.variant.is i32, %0 : !pop.variant<i8, i32>
  // CHECK: hlcf.if %[[TRUE]]
  %2 = hlcf.if %1 -> !pop.variant<f32, i8> {
    %3 = pop.variant.create %arg2 : i8 -> !pop.variant<f32, i8>
    hlcf.yield %3 : !pop.variant<f32, i8>
  } else {
    %3 = pop.variant.create %arg1 : f32 -> !pop.variant<f32, i8>
    hlcf.yield %3 : !pop.variant<f32, i8>
  }
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: i1 = <0>
  %5 = pop.variant.is f32, %2 : !pop.variant<f32, i8>
  // CHECK: return %[[FALSE]]
  kgen.return %5 : i1
}

// -----

// CHECK-LABEL: kgen.func export @known_false
kgen.func export @known_false(%arg0: i32, %arg1: i8, %arg2: i1) -> i1 {
  %0 = hlcf.if %arg2 -> !pop.variant<i8, i32, f32> {
    %1 = pop.variant.create %arg0 : i32 -> !pop.variant<i8, i32, f32>
    hlcf.yield %1 : !pop.variant<i8, i32, f32>
  } else {
    %1 = pop.variant.create %arg1 : i8 -> !pop.variant<i8, i32, f32>
    hlcf.yield %1 : !pop.variant<i8, i32, f32>
  }
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: i1 = <0>
  %2 = pop.variant.is f32, %0 : !pop.variant<i8, i32, f32>
  // CHECK: return %[[FALSE]]
  kgen.return %2 : i1
}

// -----

// CHECK-LABEL: kgen.func @always_index
// CHECK-SAME: -> index
kgen.func @always_index() -> !pop.variant<i8, index> {
  %0 = index.constant 5
  %1 = pop.variant.create %0 : index -> !pop.variant<i8, index>
  // CHECK: return %{{.*}} : index
  kgen.return %1 : !pop.variant<i8, index>
}

// CHECK-LABEL: kgen.func @entry
// CHECK-SAME: -> i32
kgen.func @entry() -> !pop.variant<i8, i32> {
  // CHECK: %[[RESULT:.*]] = kgen.call
  // CHECK-SAME: -> index
  %0 = kgen.call @always_index() : () -> !pop.variant<i8, index>
  // CHECK: pop.variant.create %[[RESULT]]
  %1 = pop.variant.is i8, %0 : !pop.variant<i8, index>
  // CHECK: %[[IF_RESULT:.*]] = hlcf.if
  %2 = hlcf.if %1 -> !pop.variant<i8, i32> {
    %3 = pop.variant.get %0 : !pop.variant<i8, index> as i8
    %4 = pop.variant.create %3 : i8 -> !pop.variant<i8, i32>
    hlcf.yield %4 : !pop.variant<i8, i32>
  } else {
    %3 = kgen.param.constant: scalar<si32> = <<0>>
    %4 = pop.cast_to_builtin %3 : !pop.simd<1, si32> to i32
    %5 = pop.variant.create %4 : i32 -> !pop.variant<i8, i32>
    hlcf.yield %5 : !pop.variant<i8, i32>
  }
  // CHECK: %[[RETURN:.*]] = pop.variant.get %[[IF_RESULT]] : !pop.variant<i8, i32> as i32
  // CHECK: return %[[RETURN]]
  kgen.return %2 : !pop.variant<i8, i32>
}

// CHECK-LABEL: kgen.func export @public
kgen.func export @public() {
  // CHECK-NEXT: @entry() : () -> i32
  %0 = kgen.call @entry() : () -> !pop.variant<i8, i32>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @do_not_rewrite
// CHECK-SAME: -> !pop.variant<index>
kgen.func @do_not_rewrite() -> !pop.variant<index> {
  %0 = index.constant 0
  %1 = pop.variant.create %0 : index -> !pop.variant<index>
  kgen.return %1 : !pop.variant<index>
}

// CHECK-LABEL: kgen.func export @call
kgen.func export @call() {
  // CHECK-NEXT: constant: () -> !pop.variant<index> = <@do_not_rewrite>
  %0 = kgen.param.constant: () -> !pop.variant<index> = <@do_not_rewrite>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @dead_code
// CHECK-SAME: -> !pop.variant<index>
kgen.func @dead_code() -> !pop.variant<index> {
  %0 = index.constant 0
  %1 = pop.variant.create %0 : index -> !pop.variant<index>
  // CHECK: return %{{.*}} : !pop.variant<index>
  kgen.return %1 : !pop.variant<index>
}

// -----

kgen.func @always_i32(%a: i32) -> !pop.variant<i32, f32> {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  kgen.return %0 : !pop.variant<i32, f32>
}

// Make sure all callsites are rewritten.

// CHECK-LABEL: kgen.func export @first_callsite
kgen.func export @first_callsite(%a: i32) {
  // CHECK: @always_i32(%arg0) : (i32) -> i32
  %0 = kgen.call @always_i32(%a) : (i32) -> !pop.variant<i32, f32>
  kgen.return
}

// CHECK-LABEL: kgen.func export @second_callsite
kgen.func export @second_callsite(%a: i32) {
  // CHECK: @always_i32(%arg0) : (i32) -> i32
  %0 = kgen.call @always_i32(%a) : (i32) -> !pop.variant<i32, f32>
  %1 = kgen.call @always_i32(%a) : (i32) -> !pop.variant<i32, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @multiple_returns
// CHECK-SAME: -> i32
kgen.func @multiple_returns(%a: i32, %b: i64) -> !pop.variant<i32, i64> {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, i64>
  %1 = hlcf.loop (%arg0 = %0 : !pop.variant<i32, i64>) -> !pop.variant<i32, i64> {
    %2 = pop.variant.is i32, %arg0 : !pop.variant<i32, i64>
    // CHECK: hlcf.if
    hlcf.if %2 {
      // CHECK-NEXT: %[[RES:.*]] = pop.variant.get %arg2 : !pop.variant<i32, i64> as i32
      // CHECK-NEXT: kgen.return %[[RES]]
      kgen.return %arg0 : !pop.variant<i32, i64>
    } else {
      hlcf.yield
    }
    // COM: This is valid because the code is dead.
    // CHECK: %[[I64:.*]] = pop.variant.create %arg1 : i64 ->
    // CHECK-NEXT: %[[I32:.*]] = pop.variant.get %[[I64]] : !pop.variant<i32, i64> as i32
    // CHECK-NEXT: kgen.return %[[I32]]
    %3 = pop.variant.create %b : i64 -> !pop.variant<i32, i64>
    kgen.return %3 : !pop.variant<i32, i64>
  }
  // CHECK: %[[RES:.*]] = pop.variant.get %1 : !pop.variant<i32, i64> as i32
  // CHECK-NEXT: kgen.return %[[RES]]
  kgen.return %1 : !pop.variant<i32, i64>
}

// CHECK-LABEL: kgen.func export @call_it
kgen.func export @call_it(%a: i32, %b: i64) {
  // CHECK: kgen.call @multiple_returns
  // CHECK-SAME: (i32, i64) -> i32
  %0 = kgen.call @multiple_returns(%a, %b) : (i32, i64) -> !pop.variant<i32, i64>
  kgen.return
}
