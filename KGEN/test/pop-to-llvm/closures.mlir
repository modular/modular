// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-pop-closures-to-llvm %s | FileCheck %s

// CHECK-LABEL: @my_fn
kgen.func @my_fn(%arg0: index, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg1 : !pop.scalar<f32>
}

// CHECK-LABEL: @partial_apply
kgen.func @partial_apply() -> !pop.closure<(index) -> !pop.scalar<f32>> {
  // CHECK:  %[[CONST:.*]] = kgen.param.constant: !pop.scalar<f32> = <<"1.20000005">>
  // CHECK:  %[[BOUNDARG:.*]] = builtin.unrealized_conversion_cast %[[CONST]] : !pop.scalar<f32> to f32
  // CHECK:  %[[FN:.*]] = kgen.addressof @my_fn : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  %0 = kgen.param.constant: !pop.scalar<f32> = <<"1.2">>
  %1 = kgen.addressof @my_fn : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  // CHECK:  %[[FNPTR:.*]] = builtin.unrealized_conversion_cast %[[FN]] : (index, !pop.scalar<f32>) -> !pop.scalar<f32> to !llvm.ptr<func<f32 (i64, f32)>>
  // CHECK:  %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
  // CHECK:  %[[WRAPPER:.*]] = llvm.mlir.addressof @closure_wrapper_fn : !llvm.ptr<func<f32 (ptr, i64)>>
  // CHECK:  %[[WRAPPERPTR:.*]] = llvm.bitcast %[[WRAPPER]] : !llvm.ptr<func<f32 (ptr, i64)>> to !llvm.ptr
  // CHECK:  %[[INSERT_WRAPPER:.*]] = llvm.insertvalue %[[WRAPPERPTR]], %[[STRUCT]][0]
  // CHECK:  %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
  // CHECK:  %[[ENV_STRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(f32)>)>
  // CHECK: %[[BOUNDARGPTR0:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 1, 0]
  // CHECK:  llvm.store %[[BOUNDARG]], %[[BOUNDARGPTR0]] : !llvm.ptr<f32>
  // CHECK: %[[CALLEEPTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<ptr<func<f32 (i64, f32)>>>
  // CHECK:  llvm.store %[[FNPTR]], %[[CALLEEPTR]] : !llvm.ptr<ptr<func<f32 (i64, f32)>>>
  // CHECK:  %[[ERASED_ENV:.*]] = llvm.bitcast %[[ENV_STRUCT]] : !llvm.ptr<struct<(ptr, struct<(f32)>)>> to !llvm.ptr
  // CHECK:  %[[INSERT:.*]] = llvm.insertvalue %[[ERASED_ENV]], %[[STRUCT]][1]
  // CHECK:  %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[STRUCT]] : !llvm.struct<(ptr, ptr)> to !pop.closure<(index) -> !pop.scalar<f32>>
  // CHECK:  kgen.return %[[CAST]] : !pop.closure<(index) -> !pop.scalar<f32>>
  %2 = pop.partial_apply %1(?, %0) : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  kgen.return %2 : !pop.closure<(index) -> !pop.scalar<f32>>
}
// CHECK: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr, %arg1: i64) -> f32 {
// CHECK:    %[[ENV_STRUCT:.*]] = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(ptr, struct<(f32)>)>>
// CHECK:    %[[CALLEE:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<func<f32 (i64, f32)>>
// CHECK:    %[[BOUNDARGPTR0:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 1, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<f32>
// CHECK:    %[[BOUNDARG0:.*]] = llvm.load %[[BOUNDARGPTR0]]
// CHECK:    %[[RESULT:.*]] = llvm.call %[[CALLEE]](%arg1, %[[BOUNDARG0]]) : (i64, f32) -> f32
// CHECK:    llvm.return %[[RESULT]] : f32
// CHECK:  }

// -----

// CHECK-LABEL: @call_indirect
kgen.func @call_indirect(%fn: (i32, i64) -> (f32, f64), %a: i32, %b: i64) -> (f32, f64) {
  // CHECK: %[[FN:.*]] = builtin.unrealized_conversion_cast %arg0 : (i32, i64) -> (f32, f64) to !llvm.ptr<func<struct<(f32, f64)> (i32, i64)>>
  // CHECK: %[[RESULT:.*]] = llvm.call %[[FN]](%arg1, %arg2) : (i32, i64) -> !llvm.struct<(f32, f64)>
  // CHECK: %[[R0:.*]] = llvm.extractvalue %[[RESULT]][0]
  // CHECK: %[[R1:.*]] = llvm.extractvalue %[[RESULT]][1]
  %0:2 = pop.call_indirect %fn(%a, %b) : (i32, i64) -> (f32, f64)
  // CHECK: return %[[R0]], %[[R1]] : f32, f64
  kgen.return %0#0, %0#1 : f32, f64
}

// -----

kgen.func @call_indirect(%fn: !pop.closure<(i32, i64) -> (f32, f64)>, %a: i32, %b: i64) -> (f32, f64) {
  // %[[CLOSURE:.*]] = builtin.unrealized_conversion_cast %arg0 : !pop.closure<(i32, i64) -> (f32, f64)> to !llvm.struct<(ptr, ptr)>
  // %[[WRAPPERFN:.*]] = llvm.extractvalue %[[CLOSURE]][0] : !llvm.struct<(ptr, ptr)>
  // %[[ENV:.*]] = llvm.extractvalue %[[CLOSURE]][0]
  // %[[WRAPPERFNCAST:.*]] = llvm.bitcast %[[WRAPPERFN]] : !llvm.ptr to !llvm.ptr<func<struct<(f32, f64)> (ptr, i32, i64)>>
  // %[[RESULTS:.*]] = llvm.call %[[WRAPPERFNCAST]](%[[ENV]], %arg1, %arg2) : (!llvm.ptr, i32, i64) -> !llvm.struct<(f32, f64)>
  // %[[RESULT0:.*]] = llvm.extractvalue %[[RESULTS]][0]
  // %[[RESULT1:.*]] = llvm.extractvalue %[[RESULTS]][1]
  // kgen.return %[[RESULT0]], %[[RESULT1]] : f32, f64
  %0:2 = pop.call_indirect %fn(%a, %b) : !pop.closure<(i32, i64) -> (f32, f64)>
  kgen.return %0#0, %0#1 : f32, f64
}

// -----

kgen.func @call_indirect_closure(%arg0: (index, f32) -> index, %arg1 : f32, %arg2 : index) -> index {
  // CHECK: %[[FN:.*]] = builtin.unrealized_conversion_cast %arg0 : (index, f32) -> index to !llvm.ptr<func<i64 (i64, f32)>>
  // CHECK: %[[INDEX:.*]] = builtin.unrealized_conversion_cast %arg2 : index to i64
  // CHECK: %[[CLOSURE:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
  // CHECK: %[[WRAPPERFN:.*]] = llvm.mlir.addressof @closure_wrapper_fn : !llvm.ptr<func<i64 (ptr, i64)>>
  // CHECK: %[[WRAPPERFNCAST:.*]] = llvm.bitcast %[[WRAPPERFN]] : !llvm.ptr<func<i64 (ptr, i64)>> to !llvm.ptr
  // CHECK: %[[INSERT0:.*]] = llvm.insertvalue %[[WRAPPERFNCAST]], %[[CLOSURE]][0]
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
  // CHECK: %[[ENVSTRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(f32)>)>
  // CHECK: %[[BOUNDARG0:.*]] = llvm.getelementptr %[[ENVSTRUCT]][0, 1, 0]
  // CHECK: llvm.store %arg1, %[[BOUNDARG0]] : !llvm.ptr<f32>
  // CHECK: %[[CALLEE:.*]] = llvm.getelementptr %[[ENVSTRUCT]][0, 0]
  // CHECK: llvm.store %[[FN]], %[[CALLEE]] : !llvm.ptr<ptr<func<i64 (i64, f32)>>>
  // CHECK: %[[ENVSTRUCTCAST:.*]] = llvm.bitcast %[[ENVSTRUCT]]
  // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[ENVSTRUCTCAST]], %[[CLOSURE]][1]
  // CHECK: %[[WRAPPERFNPTR:.*]] = llvm.extractvalue %[[CLOSURE]][0]
  // CHECK: %[[ENVPTR:.*]] = llvm.extractvalue %[[CLOSURE]][1]
  // CHECK: %[[CASTWRAPPERFNPTR:.*]] = llvm.bitcast %[[WRAPPERFNPTR]] : !llvm.ptr to !llvm.ptr<func<i64 (ptr, i64)>>
  // CHECK: %[[RESULT:.*]] = llvm.call %[[CASTWRAPPERFNPTR]](%[[ENVPTR]], %[[INDEX]]) : (!llvm.ptr, i64) -> i64
  // CHECK: %[[CASTRESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT]]
  // CHECK: kgen.return %[[CASTRESULT]] : index
  %0 = pop.partial_apply %arg0(?, %arg1) : (index, f32) -> index
  %1 = pop.call_indirect %0(%arg2) : !pop.closure<(index) -> index>
  kgen.return %1 : index
}

// CHECK: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr, %arg1: i64) -> i64 {
// CHECK:  %[[ENV:.*]] = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(ptr, struct<(f32)>)>>
// CHECK:  %[[CALLEE:.*]] = llvm.getelementptr %[[ENV]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<func<i64 (i64, f32)>>
// CHECK:  %[[BOUNDARG0PTR:.*]] = llvm.getelementptr %[[ENV]][0, 1, 0]
// CHECK:  %[[BOUNDARG0:.*]] = llvm.load %[[BOUNDARG0PTR]] : !llvm.ptr<f32>
// CHECK:  %[[RESULT:.*]] = llvm.call %[[CALLEE]](%arg1, %[[BOUNDARG0]]) : (i64, f32) -> i64
// CHECK:  llvm.return %[[RESULT]] : i64
// CHECK: }

// -----

// CHECK-LABEL: @test_lifetimes
kgen.func @test_lifetimes(%arg0: (index, f32) -> index, %arg1: f32, %cond: i1) -> () {
  // CHECK: %[[CALLEE:.*]] = builtin.unrealized_conversion_cast %arg0
  scf.if %cond {
    // CHECK: %[[CLOSURE:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    // CHECK: %[[WRAPPERFN:.*]] = llvm.mlir.addressof @closure_wrapper_fn
    // CHECK: %[[WRAPPERFNCAST:.*]] = llvm.bitcast %[[WRAPPERFN]]
    // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[WRAPPERFNCAST]], %[[CLOSURE]][0]
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK: %[[STRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(f32)>)>
    // CHECK: llvm.intr.lifetime.start 16, %[[STRUCT]] : !llvm.ptr<struct<(ptr, struct<(f32)>)>>
    // CHECK: %[[BOUNDARG0:.*]] = llvm.getelementptr %[[STRUCT]][0, 1, 0]
    // CHECK: llvm.store %arg1, %[[BOUNDARG0]]
    // CHECK: %[[CALLEEPTR:.*]] = llvm.getelementptr %[[STRUCT]][0, 0]
    // CHECK: llvm.store %[[CALLEE]], %[[CALLEEPTR]]
    // CHECK: %[[ERASEDSTRUCT:.*]] = llvm.bitcast %[[STRUCT]]
    // CHECK: %[[INSERT2:.*]] = llvm.insertvalue %[[ERASEDSTRUCT]], %[[CLOSURE]][1]
    // CHECK: %[[CONSTANT:.*]] = kgen.param.constant
    // CHECK: llvm.intr.lifetime.end 16, %[[STRUCT]]
    %0 = pop.partial_apply %arg0(?, %arg1) : (index, f32) -> index
    %1 = kgen.param.constant: !pop.scalar<f32> = <<"1.0">>
    scf.yield
  }
  kgen.return
}

// -----

kgen.func @test_name_collison(%fn0: (f32) -> (), %fn1: (f64) -> (), %arg0: f32, %arg1: f64) -> () {
  %0 = pop.partial_apply %fn0(%arg0) : (f32) -> ()
  %1 = pop.partial_apply %fn1(%arg1) : (f64) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr) {
// CHECK-LABEL: llvm.func @closure_wrapper_fn_0(%arg0: !llvm.ptr) {
