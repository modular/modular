// RUN: kgen-opt -automatic-inline -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-NOT: @inline_me.a
kgen.func @inline_me.a() always_inline {
  "inline.a"() : () -> ()
  kgen.return
}

// CHECK-NOT: @inline_me.b
kgen.func @inline_me.b() always_inline {
  "inline.b"() : () -> ()
  kgen.call @inline_me.a() : () -> ()
  kgen.return
}

// CHECK-LABEL: @top0
kgen.func @top0() {
  // CHECK-NEXT: inline.a
  // CHECK-NOT: kgen.call
  kgen.call @inline_me.a() : () -> ()
  // CHECK: label
  "label"() : () -> ()
  // CHECK-NEXT: inline.b
  // CHECK-NEXT: inline.a
  // CHECK-NOT: kgen.call
  kgen.call @inline_me.b() : () -> ()
  kgen.return
}

kgen.func @has_arg(%arg0: index) -> index always_inline {
  "use"(%arg0) : (index) -> ()
  %0 = "new"() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @top1
kgen.func @top1() -> index {
  %0 = "produce"() : () -> index
  // CHECK: "use"(%0)
  // CHECK-NOT: kgen.call
  // CHECK: %1 = "new"
  %1 = kgen.call @has_arg(%0) : (index) -> index
  // CHECK: return %1
  kgen.return %1 : index
}

kgen.func @two_returns(%a: i1, %b: index, %c: index) -> index always_inline {
  hlcf.if %a {
    kgen.return %b : index
  } else {
    hlcf.yield
  }
  kgen.return %c : index
}

// CHECK-LABEL: @top2
kgen.func @top2() -> index {
  %0:3 = "produce"() : () -> (i1, index, index)
  // CHECK: %1 = hlcf.loop
    // CHECK-NEXT: hlcf.if %0#0
      // CHECK-NEXT: hlcf.break "{{.*}}" %0#1
  %1 = kgen.call @two_returns(%0#0, %0#1, %0#2) : (i1, index, index) -> index
    // CHECK: hlcf.break "{{.*}}" %0#2
  // CHECK: return %1
  kgen.return %1 : index
}

// -----

kgen.func @async_fn(%arg0: index) async -> index always_inline {
  %0 = pop.compiler.global_load "cond" : i1
  hlcf.if %0 {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.func @call_it() -> !co.routine {
  %idx2 = index.constant 2
  %true = index.bool.constant true
  pop.compiler.global_store "cond", %true : i1
  // CHECK: %0 = co.execute : index
  // CHECK:   %1 = pop.compiler.global_load
  // CHECK:   hlcf.if %1
  // CHECK:     kgen.return %idx1
  // CHECK:   kgen.return %idx2
  %coroHdl = co.invoke[(index) async -> index: @async_fn](%idx2)
  // CHECK: kgen.return %0
  kgen.return %coroHdl : !co.routine
}

kgen.func @byref_result(%arg0: index, %arg1: !kgen.pointer<index> byref_result) async -> index always_inline {
  pop.store %arg0, %arg1 : !kgen.pointer<index>
  kgen.return %arg0: index
}

kgen.func @byref_error(%arg0: index, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) async|throws -> index always_inline {
  pop.store %arg0, %arg1 : !kgen.pointer<index>
  pop.store %arg0, %arg2 : !kgen.pointer<index>
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @call_byref
kgen.func @call_byref(%arg0: index) {
  // CHECK-NEXT: co.execute : index (%arg1: !kgen.pointer<index> byref_result)
  // CHECK-NEXT:   store %arg0, %arg1
  // CHECK-NEXT:   return %arg0
  // CHECK-NEXT: }
  co.invoke[(index, !kgen.pointer<index> byref_result) async -> index: @byref_result](%arg0)
  // CHECK-NEXT: co.execute : index (%arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result)
  // CHECK-NEXT:   store %arg0, %arg1
  // CHECK-NEXT:   store %arg0, %arg2
  // CHECK-NEXT:   return %arg0
  // CHECK-NEXT: }
  co.invoke[(index, !kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) async|throws -> index: @byref_error](%arg0)
  kgen.return
}

// -----

kgen.func @loop() always_inline {
  hlcf.loop {
    "inline.me"() : () -> ()
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @top
kgen.func @top() {
  // CHECK-NEXT: hlcf.loop
  // CHECK-NEXT: inline.me
  kgen.call @loop() : () -> ()
  kgen.return
}

// -----

kgen.func @unreachable_and_early_ret() always_inline {
  %true = index.bool.constant true
  hlcf.if %true {
    kgen.return
  } else {
    hlcf.yield
  }
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @call_it
kgen.func @call_it() {
  // CHECK-NEXT: hlcf.loop
      // CHECK: hlcf.break
    // CHECK: kgen.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: kgen.return
  kgen.call @unreachable_and_early_ret() : () -> ()
  kgen.return
}

// -----

kgen.func @capturing_closure() capturing -> index always_inline {
  %0 = pop.compiler.global_load "var" : index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  %0 = index.constant 0
  pop.compiler.global_store "var", %0 : index
  // CHECK: pop.compiler.global_load "var"
  %1 = kgen.call @capturing_closure() : () capturing -> index
  kgen.return
}

// -----

kgen.func @callee(%arg0: index, %arg1: index) capturing always_inline {
  "use"(%arg0, %arg1) : (index, index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  %idx0 = index.constant 0
  // CHECK: %0 = kgen.stage_closure = (%arg0: index) capturing
  // CHECK-NEXT: "use"(%idx0, %arg0)
  %0 = kgen.create_closure[(index, index) capturing -> (): @callee](%idx0)

  // CHECK: call_indirect %0(%idx0)
  kgen.call_indirect %0(%idx0) : (index) capturing -> ()
  kgen.return
}

// -----

kgen.func @has_closure() always_inline {
  kgen.stage_closure = () {
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  // CHECK: kgen.stage_closure
  // CHECK-NEXT: kgen.return
  kgen.call @has_closure() : () -> ()
  kgen.return
}

// -----

kgen.func @two_callers(%arg0: index, %arg1: index) always_inline {
  kgen.return
}

// CHECK: kgen.func @caller0
kgen.func @caller0() {
  %idx0 = index.constant 0
  // CHECK: stage_closure = (%arg0: index) capturing
  kgen.create_closure[(index, index) -> (): @two_callers](%idx0)
  kgen.return
}

// CHECK: kgen.func @caller1
kgen.func @caller1() {
  %idx0 = index.constant 0
  // CHECK: stage_closure = (%arg0: index) capturing
  kgen.create_closure[(index, index) -> (): @two_callers](%idx0)
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @dontinlineme
kgen.generator @dontinlineme() always_inline {
  %idx0 = index.constant 0
  kgen.return
}

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  // CHECK-NEXT: kgen.call @dontinlineme
  kgen.call @dontinlineme() : () -> ()
  kgen.return
}

// -----

kgen.func @noreturn() always_inline {
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @invoke_noreturn
kgen.func @invoke_noreturn() {
  // CHECK-NEXT: hlcf.loop
  // CHECK-NEXT: kgen.unreachable
  kgen.call @noreturn() : () -> ()
  kgen.return
}

// -----

kgen.func @wrap_source_loc_0() always_inline {
  %line, %col, %fileName = kgen.source_loc[0]
  kgen.return
}

kgen.func @wrap_source_loc_1() always_inline {
  %line, %col, %fileName = kgen.source_loc[1]
  kgen.return
}

kgen.func @test_wrap_source_loc_0() always_inline {
  kgen.call @wrap_source_loc_0() : () -> () loc("some_file.mojo":4:6)
  kgen.return
}

kgen.func @call_wrapped_source_loc_1() always_inline {
  kgen.call @wrap_source_loc_1() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @test_wrapped_source_loc_1
kgen.func @test_wrapped_source_loc_1() {
  // CHECK-DAG: kgen.param.constant = <10>
  // CHECK-DAG: kgen.param.constant = <12>
  // CHECK-DAG: kgen.param.constant: string = <"other_file.mojo">
  // CHECK-NOT: kgen.call
  kgen.call @call_wrapped_source_loc_1() : () -> () loc("other_file.mojo":10:12)
  kgen.return
}

kgen.func @test_wrapped_source_loc_1_inlined() always_inline {
  kgen.call @call_wrapped_source_loc_1() : () -> () loc("another_file.mojo":42:13)
  kgen.return
}

// CHECK-LABEL: kgen.func @test_source_loc
kgen.func @test_source_loc() {
  // CHECK-DAG: kgen.param.constant = <4>
  // CHECK-DAG: kgen.param.constant = <6>
  // CHECK-DAG: kgen.param.constant: string = <"some_file.mojo">
  kgen.call @test_wrap_source_loc_0() : () -> ()

  // CHECK-DAG: kgen.param.constant = <42>
  // CHECK-DAG: kgen.param.constant = <13>
  // CHECK-DAG: kgen.param.constant: string = <"another_file.mojo">
  kgen.call @test_wrapped_source_loc_1_inlined() : () -> ()

  kgen.return
}
