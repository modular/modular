// RUN: kgen-opt -canonicalizer -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @if_to_select
kgen.func @if_to_select(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK-NEXT: pop.select %arg0, %arg1, %arg2 : f32
  %0 = hlcf.if %arg0 -> f32 {
    hlcf.yield %arg1 : f32
  } else {
    hlcf.yield %arg2 : f32
  }
  kgen.return %0 : f32
}


// CHECK-LABEL: @if_to_select
kgen.func @if_to_select_dead_if(%arg0: i1) {
  // CHECK-NEXT: return
  hlcf.if %arg0 {
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: @if_to_select_multiple
kgen.func @if_to_select_multiple(%arg0: i1, %arg1: f32, %arg2: i32,
                                 %arg3: f32, %arg4: i32) -> (f32, i32) {
  // CHECK-NEXT: pop.select %arg0, %arg1, %arg3 : f32
  // CHECK-NEXT: pop.select %arg0, %arg2, %arg4 : i32
  %0:2 = hlcf.if %arg0 -> f32, i32 {
    hlcf.yield %arg1, %arg2 : f32, i32
  } else {
    hlcf.yield %arg3, %arg4 : f32, i32
  }
  kgen.return %0#0, %0#1 : f32, i32
}

// CHECK-LABEL: @indexify_comparison
kgen.func @indexify_comparison(%arg0: index) -> i1 {
  // CHECK: %0 = index.cmp sgt(%arg0, %idx1)
  %simd = kgen.param.constant: scalar<index> = <1>
  %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
  %1 = pop.cmp gt(%0, %simd) : !pop.scalar<index>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<bool> to i1
  // CHECK: return %0
  kgen.return %2 : i1
}

// CHECK-LABEL: @canonicalize_loop_range
kgen.func @canonicalize_loop_range(%arg0: index, %arg1: index, %arg2: index) -> (i1, i1, i1, i1) {
  // CHECK:      [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[V0:%.*]] = index.cmp slt(%arg0, %arg1)
  %0 = index.cmp slt(%arg0, %arg1)
  %idx0 = index.constant 0
  %1 = index.sub %arg1, %arg0
  %2 = pop.select %0, %1, %idx0 : index
  %3 = index.cmp sgt(%2, %idx0)

  // CHECK-NEXT: [[V1:%.*]] = index.cmp sgt(%arg1, %arg0)
  %4 = index.cmp sgt(%arg1, %arg0)
  %5 = index.sub %arg1, %arg0
  %6 = pop.select %4, %5, %idx0 : index
  %7 = index.cmp sgt(%6, %idx0)

  // CHECK-NEXT: [[V2:%.*]] = index.cmp sgt(%arg2, [[IDX0]])
  %8 = index.cmp sgt(%arg2, %idx0)
  %9 = pop.select %8, %arg2, %idx0 : index
  %10 = index.cmp sgt(%9, %idx0)

  // CHECK-NEXT: [[V3:%.*]] = index.cmp slt([[IDX0]], %arg2)
  %11 = index.cmp slt(%idx0, %arg2)
  %12 = pop.select %11, %arg2, %idx0 : index
  %13 = index.cmp sgt(%12, %idx0)

  // CHECK-NEXT: return [[V0]], [[V1]], [[V2]], [[V3]]
  kgen.return %3, %7, %10, %13 : i1, i1, i1, i1
}

// CHECK-LABEL: @condition_propagation
kgen.func @condition_propagation(%cond: i1) {
  // CHECK: hlcf.if
  hlcf.if %cond {
    // CHECK-NEXT: "use"(%true)
    "use"(%cond) : (i1) -> ()
    hlcf.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: "use"(%false)
    "use"(%cond) : (i1) -> ()
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: @not_cmp
kgen.func @not_cmp(%arg0: index, %arg1: index) -> i1 {
  %simd_0 = kgen.param.constant: scalar<bool> = <true>
  // CHECK-NEXT: %0 = index.cmp ugt(%arg0, %arg1)
  %0 = index.cmp ule(%arg0, %arg1)
  %1 = pop.cast_from_builtin %0 : i1 to !pop.scalar<bool>
  %2 = pop.xor %1, %simd_0 : <1, bool>
  %3 = pop.cast_to_builtin %2 : !pop.scalar<bool> to i1
  // CHECK-NEXT: return %0
  kgen.return %3 : i1
}

// CHECK-LABEL: kgen.func @select_variant_is_0
kgen.func @select_variant_is_0(%var: !kgen.variant<index, i64>) -> (index, i64) {
  %0 = kgen.param.constant = <*?>
  %1 = kgen.param.constant: i64 = <*?>
  %2 = kgen.variant.is %var, 0 : !kgen.variant<index, i64>
  // CHECK-NEXT: [[A:%.*]] = kgen.variant.get %arg0, 0
  // CHECK-NEXT: [[B:%.*]] = kgen.variant.get %arg0, 1
  %3 = kgen.variant.get %var, 0 : !kgen.variant<index, i64>
  %4 = kgen.variant.get %var, 1 : !kgen.variant<index, i64>
  %5 = pop.select %2, %3, %0 : index
  %6 = pop.select %2, %1, %4 : i64
  // CHECK-NEXT: return [[A]], [[B]]
  kgen.return %5, %6 : index, i64
}

// CHECK-LABEL: kgen.func @select_variant_is_1
kgen.func @select_variant_is_1(%arg0: index, %c: i1) -> (index, index) {
  // CHECK-NEXT: return %arg0, %arg0
  %0 = kgen.param.constant = <*?>
  %1 = pop.select %c, %0, %arg0 : index
  %2 = pop.select %c, %arg0, %0 : index
  kgen.return %1, %2 : index, index
}

// CHECK-LABEL: kgen.func @if_hoist_yield
kgen.func @if_hoist_yield(%arg0: i1, %arg1: index) -> (index, index) {
  %idx0 = index.constant 0
  // CHECK: hlcf.if %arg0 {
  %0 = hlcf.if %arg0 -> index {
    hlcf.yield %idx0 : index
  } else {
    kgen.unreachable
  }
  // CHECK: [[SELECT:%.*]] = pop.select %arg0, %idx0, %arg1
  // CHECK: hlcf.if %arg0 {
  %2:2 = hlcf.if %arg0 -> index, index {
    %1 = "something"() : () -> index
    hlcf.yield %1, %idx0 : index, index
  } else {
    hlcf.yield %idx0, %arg1 : index, index
  }
  // CHECK: return %idx0, [[SELECT]]
  kgen.return %0, %2#1 : index, index
}
