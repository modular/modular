// RUN: kgen-opt %s -split-input-file -prune-impossible-variants | FileCheck %s

// CHECK-LABEL: @known_true_or_false
kgen.generator public @known_true_or_false<T:type>(%arg0: i32, %arg1: f32, %arg2: i8) -> i1 {
  %0 = pop.variant.create %arg0 : i32 -> !pop.variant<T, i32>
  // CHECK: %[[TRUE:.*]] = kgen.param.constant: i1 = <1>
  %1 = pop.variant.is i32, %0 : !pop.variant<T, i32>
  // CHECK: scf.if %[[TRUE]]
  %2 = scf.if %1 -> !pop.variant<f32, i8> {
    %3 = pop.variant.create %arg2 : i8 -> !pop.variant<f32, i8>
    scf.yield %3 : !pop.variant<f32, i8>
  } else {
    %3 = pop.variant.create %arg1 : f32 -> !pop.variant<f32, i8>
    scf.yield %3 : !pop.variant<f32, i8>
  }
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: i1 = <0>
  %5 = pop.variant.is f32, %2 : !pop.variant<f32, i8>
  // CHECK: return %[[FALSE]]
  kgen.return %5 : i1
}

// -----

// CHECK-LABEL: @known_false
kgen.generator public @known_false<T:type>(%arg0: i32, %arg1: !kgen.paramref<T>, %arg2: i1) -> i1 {
  %0 = scf.if %arg2 -> !pop.variant<T, i32, f32> {
    %1 = pop.variant.create %arg0 : i32 -> !pop.variant<T, i32, f32>
    scf.yield %1 : !pop.variant<T, i32, f32>
  } else {
    %1 = pop.variant.create %arg1 : !kgen.paramref<T> -> !pop.variant<T, i32, f32>
    scf.yield %1 : !pop.variant<T, i32, f32>
  }
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: i1 = <0>
  %2 = pop.variant.is f32, %0 : !pop.variant<T, i32, f32>
  // CHECK: return %[[FALSE]]
  kgen.return %2 : i1
}

// -----

// CHECK-LABEL: @always_index
// CHECK-SAME: -> index
kgen.generator @always_index<T:type>() -> !pop.variant<T, index> {
  %0 = index.constant 5
  %1 = pop.variant.create %0 : index -> !pop.variant<T, index>
  // CHECK: return %{{.*}} : index
  kgen.return %1 : !pop.variant<T, index>
}

// CHECK-LABEL: @entry
// CHECK-SAME: -> i32
kgen.generator public @entry<T:type>() -> !pop.variant<T, i32> {
  // CHECK: %[[RESULT:.*]] = kgen.call
  // CHECK-SAME: -> index
  %0 = kgen.call @always_index<T:type = T>() : () -> !pop.variant<T, index>
  // CHECK: pop.variant.create %[[RESULT]]
  %1 = pop.variant.is !kgen.paramref<T>, %0 : !pop.variant<T, index>
  // CHECK: %[[IF_RESULT:.*]] = scf.if
  %2 = scf.if %1 -> !pop.variant<T, i32> {
    %3 = pop.variant.get %0 : !pop.variant<T, index> as !kgen.paramref<T>
    %4 = pop.variant.create %3 : !kgen.paramref<T> -> !pop.variant<T, i32>
    scf.yield %4 : !pop.variant<T, i32>
  } else {
    %3 = pop.constant(0 : si32) : !pop.scalar<si32>
    %4 = pop.cast_to_builtin %3 : !pop.scalar<si32> to i32
    %5 = pop.variant.create %4 : i32 -> !pop.variant<T, i32>
    scf.yield %5 : !pop.variant<T, i32>
  }
  // CHECK: %[[RETURN:.*]] = pop.variant.get %[[IF_RESULT]] : !pop.variant<T, i32> as i32
  // CHECK: return %[[RETURN]]
  kgen.return %2 : !pop.variant<T, i32>
}

// -----

// CHECK-LABEL: public @do_not_rewrite
// CHECK-SAME: -> !pop.variant<index>
kgen.generator public @do_not_rewrite() -> !pop.variant<index> {
  %0 = index.constant 0
  %1 = pop.variant.create %0 : index -> !pop.variant<index>
  kgen.return %1 : !pop.variant<index>
}

kgen.generator @use<fn: () -> !pop.variant<index>>() {
  kgen.return
}

kgen.generator @call() {
  kgen.call @use<fn: () -> !pop.variant<index> = @do_not_rewrite>() : () -> ()
  kgen.return
}

// -----

kgen.generator.interface @iface() -> !pop.variant<index>

// CHECK-LABEL: public @do_not_rewrite
// CHECK-SAME: -> !pop.variant<index>
kgen.generator public @do_not_rewrite() -> !pop.variant<index> implements @iface {
  %0 = index.constant 0
  %1 = pop.variant.create %0 : index -> !pop.variant<index>
  kgen.return %1 : !pop.variant<index>
}

// -----

// CHECK-LABEL: @dead_code()
// CHECK-SAME: -> !pop.variant<index>
kgen.generator module_private @dead_code() -> !pop.variant<index> {
  %0 = index.constant 0
  %1 = pop.variant.create %0 : index -> !pop.variant<index>
  // CHECK: return %{{.*}} : !pop.variant<index>
  kgen.return %1 : !pop.variant<index>
}

// -----

kgen.generator module_private @always_i32(%a: i32) -> !pop.variant<i32, f32> {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  kgen.return %0 : !pop.variant<i32, f32>
}

// CHECK-LABEL: @variant_visit
kgen.generator public @variant_visit(%a: i32, %b: i64, %c: f64) -> !pop.variant<i64, f64> {
  %0 = kgen.call @always_i32(%a) : (i32) -> !pop.variant<i32, f32>
  %1 = pop.variant.visit %0 : !pop.variant<i32, f32> -> !pop.variant<i64, f64>
  case (%v: i32) {
    %2 = pop.variant.create %b : i64 -> !pop.variant<i64, f64>
    pop.yield %2 : !pop.variant<i64, f64>
  }
  case (%v: f32) {
    %2 = pop.variant.create %c : f64 -> !pop.variant<i64, f64>
    pop.yield %2 : !pop.variant<i64, f64>
  }
  kgen.return %1 : !pop.variant<i64, f64>
}

// -----

// CHECK-LABEL: @always_i32
kgen.generator module_private @always_i32(%a: i32) -> !pop.variant<i32, f32> {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  kgen.return %0 : !pop.variant<i32, f32>
}

// CHECK-LABEL: @variant_visit
// CHECK-SAME: -> i64
kgen.generator public @variant_visit(%a: i32, %b: i64, %c: f64) -> !pop.variant<i64, f64> {
  %0 = kgen.call @always_i32(%a) : (i32) -> !pop.variant<i32, f32>
  // CHECK: %[[VISIT_RESULT:.*]] = pop.variant.visit
  %1 = pop.variant.visit %0 : !pop.variant<i32, f32> -> !pop.variant<i64, f64>
  case (%v: i32) {
    %2 = pop.variant.create %b : i64 -> !pop.variant<i64, f64>
    pop.yield %2 : !pop.variant<i64, f64>
  }
  case (%v: f32) {
    %2 = pop.variant.create %c : f64 -> !pop.variant<i64, f64>
    pop.yield %2 : !pop.variant<i64, f64>
  }
  // CHECK: %[[RESULT:.*]] = pop.variant.get %[[VISIT_RESULT]] : !pop.variant<i64, f64> as i64
  // CHECK: return %[[RESULT]] : i64
  kgen.return %1 : !pop.variant<i64, f64>
}

// -----

// CHECK-LABEL: @entry_state
// CHECK-SAME: -> index
kgen.generator public @entry_state(%a: !pop.variant<i32, f32>) -> !pop.variant<index, i1> {
  %0 = index.constant 0
  %1 = pop.variant.create %0 : index -> !pop.variant<index, i1>
  %2 = pop.variant.visit %a : !pop.variant<i32, f32> -> !pop.variant<index, i1>
  case (%v: i32) {
    pop.yield %1 : !pop.variant<index, i1>
  }
  default {
    pop.yield %1 : !pop.variant<index, i1>
  }
  kgen.return %2 : !pop.variant<index, i1>
}
