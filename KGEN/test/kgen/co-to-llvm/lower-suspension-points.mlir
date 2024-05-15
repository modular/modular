// RUN: kgen-opt %s -lower-suspension-points | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
// STUBS
llvm.func internal @anotherTask() -> !llvm.ptr {
 %nil = llvm.mlir.constant(0 : i8) : i8
 %nilptr = builtin.unrealized_conversion_cast %nil : i8 to !llvm.ptr
 llvm.return %nilptr : !llvm.ptr
}

llvm.func internal @print(%arg0: i1) {
 llvm.return
}

llvm.func internal @getElementFromContext(%continuation: !llvm.ptr) -> i1 {
  %cond = llvm.mlir.constant(0 : i1) : i1
  llvm.return %cond : i1
}
// CHECK-LABEL:  llvm.func @coro
// CHECK-NEXT:    [[STATE_SLOT:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[STATE:%.*]] = llvm.load [[STATE_SLOT]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.switch [[STATE]] : i32, ^bb1 [
// CHECK-NEXT:      1: ^bb3,
// CHECK-NEXT:      0: ^bb1
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb1:  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:    [[V2:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.cond_br [[V2]], ^bb2, ^bb4
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    [[V3:%.*]] = llvm.call @anotherTask()
// CHECK-NEXT:    [[STATE_SLOT:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[STATE:%.*]] = llvm.load [[STATE_SLOT]]
// CHECK-NEXT:    [[V6:%.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT:    [[NEW_STATE:%.*]] = llvm.add [[STATE]], [[V6]]
// CHECK-NEXT:    llvm.store [[NEW_STATE]], [[STATE_SLOT]]
// CHECK-NEXT:    [[V8:%.*]] = llvm.call @getElementFromContext([[V3]])
// CHECK-NEXT:    llvm.call @print([[V8]])
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb3:  // pred: ^bb0
// CHECK-NEXT:    [[V9:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.call @print([[V9]])
// CHECK-NEXT:    llvm.br ^bb5
// CHECK-NEXT:  ^bb4:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print([[V2]])
// CHECK-NEXT:    llvm.br ^bb5
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
llvm.func @coro(%continuation: !llvm.ptr) attributes { coro } {
  %cond = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %someContinuation = llvm.call @anotherTask() : () -> !llvm.ptr
  co.await {
    %cond1 = llvm.call @getElementFromContext(%someContinuation) : (!llvm.ptr) -> i1
    llvm.call @print(%cond1) : (i1) -> ()
	co.await.end
  }
  %cond1 = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
  llvm.call @print(%cond1) : (i1) -> ()
  llvm.br ^bb3
^bb2:
  llvm.call @print(%cond) : (i1) -> ()
  llvm.br ^bb3
^bb3:
  llvm.return
}

// CHECK-LABEL:  llvm.func @coro_multiple_suspends
// CHECK-NEXT:    [[STATE_SLOT:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[STATE:%.*]] = llvm.load [[STATE_SLOT]]
// CHECK-NEXT:    llvm.switch [[STATE]] : i32, ^bb1 [
// CHECK-NEXT:      1: ^bb3,
// CHECK-NEXT:      2: ^bb4,
// CHECK-NEXT:      0: ^bb1
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb1:  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:    [[V2:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.cond_br [[V2]], ^bb2, ^bb5
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    [[V3:%.*]] = llvm.call @anotherTask()
// CHECK-NEXT:    [[V4:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[V5:%.*]] = llvm.load [[V4]]
// CHECK-NEXT:    [[V6:%.*]] = llvm.mlir.constant(1
// CHECK-NEXT:    [[V7:%.*]] = llvm.add [[V5]], [[V6]]
// CHECK-NEXT:    llvm.store [[V7]], [[V4]]
// CHECK-NEXT:    [[V8:%.*]] = llvm.call @getElementFromContext([[V3]])
// CHECK-NEXT:    llvm.call @print([[V8]])
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb3:  // pred: ^bb0
// CHECK-NEXT:    [[V9:%.*]] = llvm.call @anotherTask() : () -> !llvm.ptr
// CHECK-NEXT:    [[V10:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[V11:%.*]] = llvm.load [[V10]]
// CHECK-NEXT:    [[V12:%.*]] = llvm.mlir.constant(1
// CHECK-NEXT:    [[V13:%.*]] = llvm.add [[V11]], [[V12]]
// CHECK-NEXT:    llvm.store [[V13]], [[V10]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb4:  // pred: ^bb0
// CHECK-NEXT:    llvm.call @getElementFromContext
// CHECK-NEXT:    llvm.call @print
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:  ^bb5:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print([[V2]]) : (i1) -> ()
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:  ^bb6:  // 2 preds: ^bb4, ^bb5
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
llvm.func @coro_multiple_suspends(%continuation: !llvm.ptr) attributes { coro } {
  %cond = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %someContinuation = llvm.call @anotherTask() : () -> !llvm.ptr
  co.await {
    %cond1 = llvm.call @getElementFromContext(%someContinuation) : (!llvm.ptr) -> i1
    llvm.call @print(%cond1) : (i1) -> ()
	co.await.end
  }
  %someContinuation2 = llvm.call @anotherTask() : () -> !llvm.ptr
  co.await {
    co.await.end
  }
  %cond1 = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
  llvm.call @print(%cond1) : (i1) -> ()
  llvm.br ^bb3
^bb2:
  llvm.call @print(%cond) : (i1) -> ()
  llvm.br ^bb3
^bb3:
  llvm.return
}
}
