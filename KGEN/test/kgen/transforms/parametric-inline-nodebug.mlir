// RUN: kgen-opt %s -inline-param=nodebug-only=true -mlir-print-debuginfo | FileCheck %s

kgen.generator @nodebug_inline_me() always_inline_no_debug {
  kgen.param.constant = <1>
  kgen.param.declare.region A = () -> () {
    kgen.return loc(#loc1)
  } loc(#loc1)
  kgen.return
}

kgen.generator @always_inline() always_inline {
  kgen.return
}

#loc1 = loc("foo.mlir":10:5)
#loc2 = loc("bar.mlir":12:7)
#locInlined = loc(callsite(#loc1 at #loc2))

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK-NEXT: kgen.param.constant = <1> loc(#[[LOC_INLINED:.*]])
  // CHECK-NEXT: kgen.param.declare.region A = () {
  // CHECK-NEXT:   kgen.return loc(#[[LOC_CALLEE:.*]])
  // CHECK-NEXT: } {isolated} loc(#[[LOC_CALLEE]])
  kgen.call @nodebug_inline_me() : () -> () loc(#locInlined)
  // CHECK-NEXT: call @always_inline
  kgen.call @always_inline() : () -> ()
  kgen.return
}

// CHECK-DAG: #[[LOC_CALLEE]] = loc("foo.mlir":10:5)
// CHECK-DAG: #[[LOC_INLINED]] = loc(unknown)
