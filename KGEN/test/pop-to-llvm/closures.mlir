// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-pop-closures-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @my_fn
  kgen.func @my_fn(%arg0: index, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
    kgen.return %arg1 : !pop.scalar<f32>
  }

  // CHECK-LABEL: @partial_apply
  kgen.func @partial_apply() -> !pop.closure<(index) -> !pop.scalar<f32>> {
    // CHECK:  %[[CONST:.*]] = kgen.param.constant: scalar<f32> = <"1.20000005">
    // CHECK:  %[[BOUNDARG:.*]] = builtin.unrealized_conversion_cast %[[CONST]] : !pop.scalar<f32> to f32
    // CHECK:  %[[FN:.*]] = kgen.addressof @my_fn : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
    // CHECK:  %[[FNPTR:.*]] = builtin.unrealized_conversion_cast %[[FN]] : (index, !pop.scalar<f32>) -> !pop.scalar<f32> to !llvm.ptr<func<f32 (i64, f32)>>
    // CHECK:  %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    // CHECK:  %[[WRAPPER:.*]] = llvm.mlir.addressof @closure_wrapper_fn : !llvm.ptr<func<f32 (ptr, i64)>>
    // CHECK:  %[[WRAPPERPTR:.*]] = llvm.bitcast %[[WRAPPER]] : !llvm.ptr<func<f32 (ptr, i64)>> to !llvm.ptr
    // CHECK:  %[[INSERT_WRAPPER:.*]] = llvm.insertvalue %[[WRAPPERPTR]], %[[STRUCT]][0]
    // CHECK:  %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK:  %[[ENV_STRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(f32)>)>
    // CHECK:  %[[BOUNDARGPTR0:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 1, 0]
    // CHECK:  llvm.store %[[BOUNDARG]], %[[BOUNDARGPTR0]] : !llvm.ptr<f32>
    // CHECK:  %[[CALLEEPTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<ptr<func<f32 (i64, f32)>>>
    // CHECK:  llvm.store %[[FNPTR]], %[[CALLEEPTR]] : !llvm.ptr<ptr<func<f32 (i64, f32)>>>
    // CHECK:  %[[ERASED_ENV:.*]] = llvm.bitcast %[[ENV_STRUCT]] : !llvm.ptr<struct<(ptr, struct<(f32)>)>> to !llvm.ptr
    // CHECK:  %[[INSERT:.*]] = llvm.insertvalue %[[ERASED_ENV]], %[[INSERT_WRAPPER]][1]
    // CHECK:  %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[INSERT]] : !llvm.struct<(ptr, ptr)> to !pop.closure<(index) -> !pop.scalar<f32>>
    // CHECK:  kgen.return %[[CAST]] : !pop.closure<(index) -> !pop.scalar<f32>>
    %0 = kgen.param.constant: scalar<f32> = <<"1.2">>
    %1 = kgen.addressof @my_fn : (index, !pop.scalar<f32>) -> !pop.scalar<f32>%2 = pop.partial_apply %1(?, %0) : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
    kgen.return %2 : !pop.closure<(index) -> !pop.scalar<f32>>
  }

  // CHECK:  llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr, %arg1: i64) -> f32 {
  // CHECK:   %[[ENV_STRUCT:.*]] = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(ptr, struct<(f32)>)>>
  // CHECK:   %[[CALLEEPTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<ptr<func<f32 (i64, f32)>>>
  // CHECK:   %[[CALLEE:.*]] = llvm.load %[[CALLEEPTR]] : !llvm.ptr<ptr<func<f32 (i64, f32)>>>
  // CHECK:   %[[BOUNDARGPTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 1, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<f32>
  // CHECK:   %[[BOUNDARG0:.*]] = llvm.load %[[BOUNDARGPTR]] : !llvm.ptr<f32>
  // CHECK:   %[[RESULT:.*]] = llvm.call %[[CALLEE]](%arg1, %[[BOUNDARG0]]) : !llvm.ptr<func<f32 (i64, f32)>>, (i64, f32) -> f32
  // CHECK:   llvm.return %[[RESULT]] : f32
  // CHECK:  }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @my_fn0
  kgen.func @my_fn0(%arg0: index, %arg1: !pop.scalar<f32>) -> () {
    kgen.return
  }

  // CHECK-LABEL: @test_closure_no_result
  kgen.func @test_closure_no_result(%arg0: index, %arg1: !pop.scalar<f32>) -> () {
    // %[[ARG0:.*]] = builtin.unrealized_conversion_cast %arg0 : index to i64
    // %[[ARG1:.*]] = builtin.unrealized_conversion_cast %arg1 : !pop.scalar<f32> to f32
    // %[[FN:.*]] = kgen.addressof @my_fn0 : (index, !pop.scalar<f32>) -> ()
    // %[[FNPTR:.*]] = builtin.unrealized_conversion_cast %[[FN]] : (index, !pop.scalar<f32>) -> () to !llvm.ptr<func<void (i64, f32)>>
    // %[[CLOSURESTRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    // %[[WRAPPERFNPTR:.*]] = llvm.mlir.addressof @closure_wrapper_fn : !llvm.ptr<func<void (ptr)>>
    // %[[ERASEDWRAPPERFNPTR:.*]] = llvm.bitcast %[[WRAPPERFNPTR]] : !llvm.ptr<func<void (ptr)>> to !llvm.ptr
    // %[[INSERT0:.*]] = llvm.insertvalue %[[ERASEDWRAPPERFNPTR]], %[[CLOSURESTRUCT]][0] : !llvm.struct<(ptr, ptr)>
    // %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
    // %[[ENVSTRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(i64, f32)>)> : (i8) -> !llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>
    // llvm.intr.lifetime.start 24, %[[ENVSTRUCT]] : !llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>
    // %[[BOUNDARG0PTR:.*]] = llvm.getelementptr %[[ENVSTRUCT]][0, 1, 0] : (!llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>) -> !llvm.ptr<i64>
    // llvm.store %[[ARG0]], %[[BOUNDARG0PTR]] : !llvm.ptr<i64>
    // %[[BOUNDARG1PTR:.*]] = llvm.getelementptr %[[ENVSTRUCT]][0, 1, 1] : (!llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>) -> !llvm.ptr<f32>
    // llvm.store %[[ARG1]], %[[BOUNDARG1PTR]] : !llvm.ptr<f32>
    // %[[CALLEEPTR:.*]] = llvm.getelementptr %[[ENVTSTRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>) -> !llvm.ptr<ptr<func<void (i64, f32)>>>
    // llvm.store %[[FNPTR]], %[[CALLEEPTR]] : !llvm.ptr<ptr<func<void (i64, f32)>>>
    // %[[ERASED_ENVSTRUCT:.*]] = llvm.bitcast %[[ENVSTRUCT]] : !llvm.ptr<struct<(ptr, struct<(i64, f32)>)>> to !llvm.ptr
    // %14 = llvm.insertvalue %[[ERASED_ENVSTRUCT]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr)>
    // llvm.intr.lifetime.end 24, %[[ENVSTRUCT]] : !llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>
    // kgen.return
    %fn = kgen.addressof @my_fn0 : (index, !pop.scalar<f32>) -> ()
    %0 = pop.partial_apply %fn(%arg0, %arg1) : (index, !pop.scalar<f32>) -> ()
    // CHECK: kgen.return
    kgen.return
  }

  // CHECK: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr) {
  // CHECK:   %[[ENV_STRUCT:.*]] = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>
  // CHECK:   %[[CALLEEPTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>) -> !llvm.ptr<ptr<func<void (i64, f32)>>>
  // CHECK:   %[[CALLEE:.*]] = llvm.load %[[CALLEEPTR]] : !llvm.ptr<ptr<func<void (i64, f32)>>>
  // CHECK:   %[[BOUNDARG0PTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 1, 0] : (!llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>) -> !llvm.ptr<i64>
  // CHECK:   %[[BOUNDARG0:.*]] = llvm.load %[[BOUNDARG0PTR]] : !llvm.ptr<i64>
  // CHECK:   %[[BOUNDARG1PTR:.*]] = llvm.getelementptr %[[ENV_STRUCT]][0, 1, 1] : (!llvm.ptr<struct<(ptr, struct<(i64, f32)>)>>) -> !llvm.ptr<f32>
  // CHECK:   %[[BOUNDARG1:.*]] = llvm.load %[[BOUNDARG1PTR]] : !llvm.ptr<f32>
  // CHECK:   llvm.call %[[CALLEE]](%[[BOUNDARG0]], %[[BOUNDARG1]]) : !llvm.ptr<func<void (i64, f32)>>, (i64, f32) -> ()
  // CHECK:   llvm.return
  // CHECK: }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @call_indirect
  kgen.func @call_indirect(%fn: (i32, i64) -> (f32, f64), %a: i32, %b: i64) -> (f32, f64) {
    // CHECK: %[[FN:.*]] = builtin.unrealized_conversion_cast %arg0 : (i32, i64) -> (f32, f64) to !llvm.ptr<func<struct<(f32, f64)> (i32, i64)>>
    // CHECK: %[[RESULT:.*]] = llvm.call %[[FN]](%arg1, %arg2) : !llvm.ptr<func<struct<(f32, f64)> (i32, i64)>>, (i32, i64) -> !llvm.struct<(f32, f64)>
    // CHECK: %[[R0:.*]] = llvm.extractvalue %[[RESULT]][0]
    // CHECK: %[[R1:.*]] = llvm.extractvalue %[[RESULT]][1]
    %0:2 = pop.call_indirect %fn(%a, %b) : (i32, i64) -> (f32, f64)
    // CHECK: return %[[R0]], %[[R1]] : f32, f64
    kgen.return %0#0, %0#1 : f32, f64
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
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
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
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
    // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[ENVSTRUCTCAST]], %[[INSERT0]][1]
    // CHECK: %[[WRAPPERFNPTR:.*]] = llvm.extractvalue %[[INSERT1]][0]
    // CHECK: %[[ENVPTR:.*]] = llvm.extractvalue %[[INSERT1]][1]
    // CHECK: %[[CASTWRAPPERFNPTR:.*]] = llvm.bitcast %[[WRAPPERFNPTR]] : !llvm.ptr to !llvm.ptr<func<i64 (ptr, i64)>>
    // CHECK: %[[RESULT:.*]] = llvm.call %[[CASTWRAPPERFNPTR]](%[[ENVPTR]], %[[INDEX]]) : !llvm.ptr<func<i64 (ptr, i64)>>, (!llvm.ptr, i64) -> i64
    // CHECK: %[[CASTRESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT]]
    // CHECK: kgen.return %[[CASTRESULT]] : index
    %0 = pop.partial_apply %arg0(?, %arg1) : (index, f32) -> index
    %1 = pop.call_indirect %0(%arg2) : !pop.closure<(index) -> index>
    kgen.return %1 : index
  }

  // CHECK: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr, %arg1: i64) -> i64 {
  // CHECK:  %[[ENV:.*]] = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(ptr, struct<(f32)>)>>
  // CHECK:  %[[CALLEEPTR:.*]] = llvm.getelementptr %[[ENV]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32)>)>>) -> !llvm.ptr<ptr<func<i64 (i64, f32)>>>
  // CHECK:  %[[CALLEE:.*]] = llvm.load %[[CALLEEPTR]] : !llvm.ptr<ptr<func<i64 (i64, f32)>>>
  // CHECK:  %[[BOUNDARG0PTR:.*]] = llvm.getelementptr %[[ENV]][0, 1, 0]
  // CHECK:  %[[BOUNDARG0:.*]] = llvm.load %[[BOUNDARG0PTR]] : !llvm.ptr<f32>
  // CHECK:  %[[RESULT:.*]] = llvm.call %[[CALLEE]](%arg1, %[[BOUNDARG0]]) : !llvm.ptr<func<i64 (i64, f32)>>, (i64, f32) -> i64
  // CHECK:  llvm.return %[[RESULT]] : i64
  // CHECK: }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @test_lifetimes
  kgen.func @test_lifetimes(%arg0: (index, f32) -> index, %arg1: f32, %cond: i1) -> () {
    // CHECK: %[[CALLEE:.*]] = builtin.unrealized_conversion_cast %arg0
    hlcf.if %cond {
      // CHECK: %[[CLOSURE:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
      // CHECK: %[[WRAPPERFN:.*]] = llvm.mlir.addressof @closure_wrapper_fn
      // CHECK: %[[WRAPPERFNCAST:.*]] = llvm.bitcast %[[WRAPPERFN]]
      // CHECK: %[[INSERT0:.*]] = llvm.insertvalue %[[WRAPPERFNCAST]], %[[CLOSURE]][0]
      // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
      // CHECK: %[[STRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(f32)>)>
      // CHECK: llvm.intr.lifetime.start 16, %[[STRUCT]] : !llvm.ptr<struct<(ptr, struct<(f32)>)>>
      // CHECK: %[[BOUNDARG0:.*]] = llvm.getelementptr %[[STRUCT]][0, 1, 0]
      // CHECK: llvm.store %arg1, %[[BOUNDARG0]]
      // CHECK: %[[CALLEEPTR:.*]] = llvm.getelementptr %[[STRUCT]][0, 0]
      // CHECK: llvm.store %[[CALLEE]], %[[CALLEEPTR]]
      // CHECK: %[[ERASEDSTRUCT:.*]] = llvm.bitcast %[[STRUCT]]
      // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[ERASEDSTRUCT]], %[[INSERT0]][1]
      // CHECK: %[[CONSTANT:.*]] = kgen.param.constant
      // CHECK: llvm.intr.lifetime.end 16, %[[STRUCT]]
      %0 = pop.partial_apply %arg0(?, %arg1) : (index, f32) -> index
      %1 = kgen.param.constant: scalar<f32> = <<"1.0">>
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @test_name_collison(%fn0: (f32) -> (), %fn1: (f64) -> (), %arg0: f32, %arg1: f64) -> () {
    %0 = pop.partial_apply %fn0(%arg0) : (f32) -> ()
    %1 = pop.partial_apply %fn1(%arg1) : (f64) -> ()
    kgen.return
  }

  // CHECK-LABEL: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr) {
  // CHECK-LABEL: llvm.func @closure_wrapper_fn_0(%arg0: !llvm.ptr) {
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @nested_closures_basic(%closure: !pop.closure<(f32) -> f32>, %arg0: f32) -> () {
    // CHECK: %[[INNER_CLOSURE:.*]] = builtin.unrealized_conversion_cast %arg0 : !pop.closure<(f32) -> f32> to !llvm.struct<(ptr, ptr)>
    // CHECK: %[[OUTER_CLOSURE:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    // CHECK: %[[O_WRAPPERFNPTR:.*]] = llvm.mlir.addressof @closure_wrapper_fn : !llvm.ptr<func<f32 (ptr)>>
    // CHECK: %[[O_WRAPPERFNPTR_ERASED:.*]] = llvm.bitcast %[[O_WRAPPERFNPTR]] : !llvm.ptr<func<f32 (ptr)>> to !llvm.ptr
    // CHECK: %[[INSERT0:.*]] = llvm.insertvalue %[[O_WRAPPERFNPTR_ERASED]], %[[OUTER_CLOSURE]][0] : !llvm.struct<(ptr, ptr)>
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK: %[[O_ENVSTRUCT:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)> : (i8) -> !llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>
    // CHECK: llvm.intr.lifetime.start 24, %6 : !llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>
    // CHECK: %[[BOUNDARG0PTR:.*]] = llvm.getelementptr %[[O_ENVSTRUCT]][0, 1, 0] : (!llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>) -> !llvm.ptr<f32>
    // CHECK: llvm.store %arg1, %[[BOUNDARG0PTR]] : !llvm.ptr<f32>
    // CHECK: %[[I_ENVSTRUCT:.*]] = llvm.extractvalue %[[INNER_CLOSURE]][1] : !llvm.struct<(ptr, ptr)>
    // CHECK: %[[CALLEEPTR:.*]] = llvm.getelementptr %[[O_ENVSTRUCT]][0, 1, 1] : (!llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>) -> !llvm.ptr<ptr>
    // CHECK: llvm.store %[[I_ENVSTRUCT]], %[[CALLEEPTR]] : !llvm.ptr<ptr>
    // CHECK: %[[I_WRAPPERFNPTR:.*]] = llvm.extractvalue %[[INNER_CLOSURE]][0] : !llvm.struct<(ptr, ptr)>
    // CHECK: %[[I_WRAPPERFNPTRCAST:.*]] = llvm.bitcast %[[I_WRAPPERFNPTR:.*]] : !llvm.ptr to !llvm.ptr<func<f32 (ptr<struct<(ptr, ptr)>>, f32)>>
    // CHECK: %[[I_WRAPPERFNPTRPTR:.*]] = llvm.getelementptr %[[O_ENVSTRUCT]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>) -> !llvm.ptr<ptr<func<f32 (ptr<struct<(ptr, ptr)>>, f32)>>>
    // CHECK: llvm.store %[[I_WRAPPERFNPTRCAST]], %[[I_WRAPPERFNPTRPTR]] : !llvm.ptr<ptr<func<f32 (ptr<struct<(ptr, ptr)>>, f32)>>>
    // CHECK: %[[O_ENVSTRUCT_ERASED:.*]] = llvm.bitcast %[[O_ENVSTRUCT]] : !llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>> to !llvm.ptr
    // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[O_ENVSTRUCT_ERASED]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr)>
    // CHECK: llvm.intr.lifetime.end 24, %[[O_ENVSTRUCT]] : !llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>
    // CHECK: kgen.return

    %0 = pop.partial_apply %closure(%arg0) : !pop.closure<(f32) -> f32>
    kgen.return
  }

  // CHECK: llvm.func @closure_wrapper_fn(%arg0: !llvm.ptr) -> f32 {
  //   CHECK: %[[CLOSURE_STRUCTPTR:.*]] = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>
  //   CHECK: %[[I_WRAPPERFNPTRPTR:.*]] = llvm.getelementptr %[[CLOSURE_STRUCTPTR]][0, 0] : (!llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>) -> !llvm.ptr<ptr<func<f32 (ptr<struct<(ptr, ptr)>>, f32)>>>
  //   CHECK: %[[I_WRAPPERFNPTR:.*]] = llvm.load %[[I_WRAPPERFNPTRPTR]] : !llvm.ptr<ptr<func<f32 (ptr<struct<(ptr, ptr)>>, f32)>>>
  //   CHECK: %[[I_ENVSTRUCT_PTRPTR:.*]] = llvm.getelementptr %[[CLOSURE_STRUCTPTR]][0, 1, 1] : (!llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>) -> !llvm.ptr<ptr<struct<(ptr, ptr)>>>
  //   CHECK: %[[I_ENVSTRUCT_PTR:.*]] = llvm.load %[[I_ENVSTRUCT_PTRPTR]] : !llvm.ptr<ptr<struct<(ptr, ptr)>>>
  //   CHECK: %[[BOUNDARG0_PTR:.*]] = llvm.getelementptr %[[CLOSURE_STRUCTPTR]][0, 1, 0] : (!llvm.ptr<struct<(ptr, struct<(f32, ptr<struct<(ptr, ptr)>>)>)>>) -> !llvm.ptr<f32>
  //   CHECK: %[[BOUNDARG0:.*]] = llvm.load %[[BOUNDARG0_PTR]] : !llvm.ptr<f32>
  //   CHECK: %[[RES:.*]] = llvm.call %[[I_WRAPPERFNPTR]](%[[I_ENVSTRUCT_PTR]], %[[BOUNDARG0]]) : !llvm.ptr<func<f32 (ptr<struct<(ptr, ptr)>>, f32)>>, (!llvm.ptr<struct<(ptr, ptr)>>, f32) -> f32
  //   CHECK: llvm.return %[[RES]] : f32
}
