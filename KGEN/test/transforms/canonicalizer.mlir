// RUN: kgen-opt -canonicalizer %s | FileCheck %s

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
