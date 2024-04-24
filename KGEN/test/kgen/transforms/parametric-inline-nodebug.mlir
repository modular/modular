// RUN: kgen-opt %s -inline-param=nodebug-only=true -mlir-print-debuginfo | FileCheck %s

kgen.generator @wrap_source_loc_0() -> !kgen.none always_inline_no_debug {
  %line, %col, %fileName = kgen.source_loc[0]
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @wrap_source_loc_1() -> !kgen.none always_inline_no_debug {
  %line, %col, %fileName = kgen.source_loc[1]
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrap_source_loc_0
kgen.generator @test_wrap_source_loc_0() -> !kgen.none always_inline_no_debug {
  // CHECK-DAG: kgen.param.constant = <4>
  // CHECK-DAG: kgen.param.constant = <6>
  // CHECK-DAG: kgen.param.constant: string = <"some_file.mojo">
  // CHECK-NOT: kgen.call
  %0 = kgen.call @wrap_source_loc_0() : () -> !kgen.none loc("some_file.mojo":4:6)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @call_wrapped_source_loc_1
kgen.generator @call_wrapped_source_loc_1() -> !kgen.none always_inline_no_debug {
  // CHECK: kgen.source_loc[0]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @wrap_source_loc_1() : () -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrapped_source_loc_1
kgen.generator @test_wrapped_source_loc_1() -> !kgen.none {
  // CHECK-DAG: kgen.param.constant = <10>
  // CHECK-DAG: kgen.param.constant = <12>
  // CHECK-DAG: kgen.param.constant: string = <"other_file.mojo">
  // CHECK-NOT: kgen.call
  %0 = kgen.call @call_wrapped_source_loc_1() : () -> !kgen.none loc("other_file.mojo":10:12)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrapped_source_loc_1_inlined
kgen.generator @test_wrapped_source_loc_1_inlined() -> !kgen.none always_inline_no_debug {
  // CHECK-DAG: kgen.param.constant = <42>
  // CHECK-DAG: kgen.param.constant = <13>
  // CHECK-DAG: kgen.param.constant: string = <"another_file.mojo">
  // CHECK-NOT: kgen.call
  %0 = kgen.call @call_wrapped_source_loc_1() : () -> !kgen.none loc("another_file.mojo":42:13)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

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
