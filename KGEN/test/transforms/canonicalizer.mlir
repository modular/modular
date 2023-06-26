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
kgen.func @canonicalize_loop_range(%arg0: index, %arg1: index) -> i1 {
  // CHECK-NEXT: %0 = index.cmp slt(%arg0, %arg1)
  %0 = index.cmp slt(%arg0, %arg1)
  %idx0 = index.constant 0
  %1 = index.sub %arg1, %arg0
  %2 = pop.select %0, %1, %idx0 : index
  %3 = index.cmp sgt(%2, %idx0)
  // CHECK-NEXT: return %0
  kgen.return %3 : i1
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
