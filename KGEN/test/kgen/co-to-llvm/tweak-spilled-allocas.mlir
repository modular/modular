// RUN: kgen-opt -tweak-spilled-allocas %s | FileCheck %s

// CHECK-LABEL: llvm.func @normal_alloca
llvm.func @normal_alloca() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: %1 = llvm.alloca
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    // CHECK-NEXT: llvm.intr.lifetime.start 4, %1
    llvm.intr.lifetime.start 4, %1 : !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i32
    // CHECK: llvm.intr.lifetime.end 4, %1
    llvm.intr.lifetime.end 4, %1 : !llvm.ptr
    // CHECK-NEXT: hlcf.break
    hlcf.break
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @spilled_alloca
llvm.func @spilled_alloca() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: %1 = llvm.alloca
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    // CHECK-NOT: llvm.intr.lifetime.start 4, %1
    llvm.intr.lifetime.start 4, %1 : !llvm.ptr
    hlcf.loop {
      co.suspend (%hdl) {
        co.suspend.end
      }
      hlcf.break
    }
    %2 = llvm.load %1 : !llvm.ptr -> i32
    // CHECK-NOT: llvm.intr.lifetime.end 4, %1
    llvm.intr.lifetime.end 4, %1 : !llvm.ptr
    hlcf.break
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @not_spilled_alloca
llvm.func @not_spilled_alloca() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: hlcf.loop
  hlcf.loop {
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: %1 = llvm.alloca
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    // CHECK-NEXT: llvm.intr.lifetime.start 4, %1
    llvm.intr.lifetime.start 4, %1 : !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i32
    // CHECK: llvm.intr.lifetime.end 4, %1
    llvm.intr.lifetime.end 4, %1 : !llvm.ptr
    // CHECK-NEXT: co.suspend
    co.suspend (%hdl) {
      co.suspend.end
    }
    hlcf.break
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @remove_alloca_from_frame
llvm.func @remove_alloca_from_frame(%cond: i1) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  // CHECK: hlcf.if
  hlcf.if %cond {
    // CHECK-NEXT: co.suspend
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: %1 = llvm.alloca
    // CHECK-NEXT: llvm.intr.lifetime.start 4, %1
    llvm.intr.lifetime.start 4, %1 : !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i32
    llvm.intr.lifetime.end 4, %1 : !llvm.ptr
    hlcf.yield
  } else {
    hlcf.yield
  }
  llvm.return
}
