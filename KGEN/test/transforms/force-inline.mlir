// RUN: kgen-opt -force-inline=update-debug-info=true -allow-unregistered-dialect -mlir-print-debuginfo -split-input-file %s | FileCheck %s

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

kgen.func @inline_me.a() always_inline {
  "inline.a"() : () -> ()
  kgen.return
}

// CHECK: kgen.func @top0
kgen.func @top0() {
  // CHECK-NEXT: inline.a
  // CHECK-SAME: loc(#[[INLINED_LOC:.*]])
  kgen.call @inline_me.a() : () -> ()
  kgen.return
}

// CHECK: #[[INLINED_LOC]] = loc(callsite(#{{.*}} at #{{.*}}))

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1
> : !debuginfo.unresolved<index>

kgen.func @nodebug_inline_me(%arg0: index) -> index always_inline_no_debug {
  %0 = index.add %arg0, %arg0
  debuginfo.value #local_variable = %arg0 : index
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func @call_it
kgen.func @call_it() -> index {
  %0 = index.constant 3
  // CHECK: index.add %idx3, %idx3 loc(#[[NODEBUG_LOC:.*]])
  // CHECK-NOT: debuginfo.value
  %1 = kgen.call @nodebug_inline_me(%0) : (index) -> index
  kgen.return %1 : index
}

// CHECK: #[[NODEBUG_LOC]] = loc("within split

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
kgen.func @call_it() -> !pop.coroutine<() -> (index)> {
  %idx2 = index.constant 2
  %true = index.bool.constant true
  pop.compiler.global_store "cond", %true : i1
  // CHECK: %0 = lit.async.execute <() -> index>
  // CHECK:   %1 = pop.compiler.global_load
  // CHECK:   hlcf.if %1
  // CHECK:     lit.async.return %idx1
  // CHECK:   lit.async.return %idx2
  %coroHdl = lit.async.call[(index) async -> index: @async_fn](%idx2)
  // CHECK: kgen.return %0
  kgen.return %coroHdl : !pop.coroutine<() -> (index)>
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

kgen.func @fat_closure() capturing -> index {
  %0 = pop.compiler.global_load "var" : index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  %0 = index.constant 0
  pop.compiler.global_store "var", %0 : index
  // CHECK: pop.compiler.global_load "var"
  %1 = kgen.call @fat_closure() : () capturing -> index
  kgen.return
}
