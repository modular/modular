// RUN: index-opt %s | index-opt | FileCheck %s

// CHECK-LABEL: func @loop
func.func @loop(%arg0: i32, %arg1: i64) {
  // CHECK: hlcf.loop {
  hlcf.loop {
    hlcf.break
  }

  // CHECK: hlcf.loop {
  hlcf.loop () -> () {
    hlcf.break
  }

  // CHECK: hlcf.loop (%{{.*}} = %arg0 : i32) {
  hlcf.loop (%0 = %arg0 : i32) -> () {
    hlcf.break
  }

  // CHECK: %{{.*}} = hlcf.loop () -> index {
  %0 = hlcf.loop () -> index {
    hlcf.continue
  }

  // CHECK: %{{.*}}:2 = hlcf.loop () -> (index, index) {
  %1:2 = hlcf.loop () -> (index, index) {
    hlcf.continue
  }

  return
}

// CHECK-LABEL: func.func @if
func.func @if(%arg0: i1, %arg1: i32, %arg2: i64) {
  // CHECK-NEXT: hlcf.if %arg0 {
  hlcf.if %arg0 {
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  // CHECK-NEXT: }
  }

  // CHECK: %{{.*}} = hlcf.if %arg0 -> i32 {
  %0 = hlcf.if %arg0 -> i32 {
    // CHECK-NEXT: hlcf.yield %arg1 : i32
    hlcf.yield %arg1 : i32
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hlcf.yield %arg1 : i32
    hlcf.yield %arg1 : i32
  }

  // CHECK: %{{.*}} = hlcf.if %arg0 -> i32, i64
  %1:2 = hlcf.if %arg0 -> i32, i64 {
    // CHECK-NEXT: hlcf.yield %arg1, %arg2 : i32, i64
    hlcf.yield %arg1, %arg2 : i32, i64
  } else {
    hlcf.yield %arg1, %arg2 : i32, i64
  }

  return
}
