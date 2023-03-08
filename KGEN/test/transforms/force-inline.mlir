// RUN: kgen-opt -force-inline -allow-unregistered-dialect -mlir-print-debuginfo -split-input-file %s | FileCheck %s

kgen.func @inline_me.a() always_inline {
  "inline.a"() : () -> ()
  kgen.return
}

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

// CHECK-LABEL: kgen.func @inline_me.a
kgen.func @inline_me.a() always_inline {
  // CHECK-NEXT: inline.a
  // CHECK-SAME: loc(#[[CALLEE_LOC:.*]])
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

// CHECK: #[[INLINED_LOC]] = loc(callsite(#[[CALLEE_LOC]] at #{{.*}}))

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
