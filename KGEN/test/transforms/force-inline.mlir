// RUN: kgen-opt -force-inline -allow-unregistered-dialect -mlir-print-debuginfo -split-input-file %s | FileCheck %s

kgen.func @inline_me.a() force_inline {
  "inline.a"() : () -> ()
  kgen.return
}

kgen.func @inline_me.b() force_inline {
  "inline.b"() : () -> ()
  kgen.call @inline_me.a() : () force_inline -> ()
  kgen.return
}

// CHECK-LABEL: @top0
kgen.func @top0() {
  // CHECK-NEXT: inline.a
  // CHECK-NOT: kgen.call
  kgen.call @inline_me.a() : () force_inline -> ()
  // CHECK: label
  "label"() : () -> ()
  // CHECK-NEXT: inline.b
  // CHECK-NEXT: inline.a
  // CHECK-NOT: kgen.call
  kgen.call @inline_me.b() : () force_inline -> ()
  kgen.return
}

kgen.func @has_arg(%arg0: index) force_inline -> index {
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
  %1 = kgen.call @has_arg(%0) : (index) force_inline -> index
  // CHECK: return %1
  kgen.return %1 : index
}

kgen.func @two_returns(%a: i1, %b: index, %c: index) force_inline -> index {
  hlcf.if %a {
    hlcf.return %b : index
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
  %1 = kgen.call @two_returns(%0#0, %0#1, %0#2) : (i1, index, index) force_inline -> index
    // CHECK: hlcf.break "{{.*}}" %0#2
  // CHECK: return %1
  kgen.return %1 : index
}

// -----

// CHECK-LABEL: kgen.func @inline_me.a
kgen.func @inline_me.a() force_inline {
  // CHECK-NEXT: inline.a
  // CHECK-SAME: loc(#[[CALLEE_LOC:.*]])
  "inline.a"() : () -> ()
  kgen.return
}

// CHECK: kgen.func @top0
kgen.func @top0() {
  // CHECK-NEXT: inline.a
  // CHECK-SAME: loc(#[[INLINED_LOC:.*]])
  kgen.call @inline_me.a() : () force_inline -> ()
  kgen.return
}

// CHECK: #[[INLINED_LOC]] = loc(callsite(#[[CALLEE_LOC]] at #{{.*}}))
