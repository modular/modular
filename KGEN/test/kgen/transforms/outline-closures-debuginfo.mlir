// RUN: kgen-opt %s -split-input-file -outline-closures=debug-build=true -mlir-print-debuginfo | FileCheck %s


// CHECK-LABEL: kgen.generator @foo_NestedClosure() -> !pop.array<0, i32> {
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i32> = <[]> loc(#[[LOC_NESTED:loc[0-9]*]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i32> loc(#[[LOC_NESTED]])
// CHECK-NEXT:  } loc(#[[LOC_NESTED]])

// CHECK-LABEL: kgen.generator @foo_Closure() -> !pop.array<0, i8> {
// CHECK-NEXT:    kgen.param.declare NestedClosure: () -> !pop.array<0, i32> = <@foo_NestedClosure> loc(#[[LOC_NESTED_DEC:loc[0-9]*]])
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i8> = <[]> loc(#[[LOC_CLOSURE:loc[0-9]*]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i8> loc(#[[LOC_CLOSURE]])
// CHECK-NEXT:  } loc(#[[LOC_CLOSURE]])

// CHECK-LABEL: kgen.generator @foo_OtherClosure() always_inline_no_debug {
// CHECK-NEXT:    kgen.return loc(#[[LOC1:.*]])
// CHECK-NEXT:  } loc(#[[LOC1]])

// CHECK-LABEL: kgen.generator @foo_NestedCapturing()
// CHECK-NEXT:    %0 = pop.compiler.global_load "foo_context_var_0"

// CHECK-LABEL: kgen.generator @foo_Capturing()
// CHECK-NEXT:    %0 = pop.compiler.global_load "foo_context_var_1"
// CHECK-NEXT:    pop.compiler.global_store "foo_context_var_0", %0

// CHECK-LABEL: kgen.generator @foo(
// CHECK-SAME:      %[[ARG:.*]]: index
// CHECK-NEXT:    kgen.param.declare Closure: () -> !pop.array<0, i8> = <@foo_Closure> loc(#[[LOC_CLOSURE_DEC:.*]])
// CHECK-NEXT:    kgen.param.declare OtherClosure: () -> () = <@foo_OtherClosure> loc(#[[LOC_FOO:.*]])
// CHECK-NEXT:    pop.compiler.global_store "foo_context_var_1", %[[ARG]] : index loc(#[[LOC_CAP:.*]])
// CHECK-NEXT:    kgen.param.declare Capturing: () capturing -> () = <@foo_Capturing> loc(#[[LOC_CAP]])
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[LOC_FOO]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i1> loc(#[[LOC_FOO]])
// CHECK-NEXT:  } loc(#[[LOC_FOO]])

kgen.generator @foo(%arg0: index) -> !pop.array<0, i1> {
  kgen.param.declare.region Closure = () -> !pop.array<0, i8> {
    kgen.param.declare.region NestedClosure = () -> !pop.array<0, i32> {
      %array_3 = kgen.param.constant: array<0, i32> = <[]> loc(#locNested)
      kgen.return %array_3 : !pop.array<0, i32> loc(#locNested)
    } loc(#locNested)

    %array_2 = kgen.param.constant: array<0, i8> = <[]> loc(#locClosure)
    kgen.return %array_2 : !pop.array<0, i8> loc(#locClosure)
  } loc(#locClosure)

  kgen.param.declare.region OtherClosure = () -> () always_inline_no_debug {
    kgen.return loc(#loc1)
  } loc(#loc1)

  kgen.param.declare.region Capturing = () capturing {
    kgen.param.declare.region NestedCapturing = () capturing -> index {
      kgen.return %arg0 : index loc(#locNestedCap)
    } loc(#locNestedCap)
    kgen.return loc(#locCap)
  } loc(#locCap)

  %array = kgen.param.constant: array<0, i1> = <[]> loc(#locFoo)
  kgen.return %array : !pop.array<0, i1> loc(#locFoo)
} loc(#locFoo)

// CHECK-DAG: #[[LOC1]] = loc("foo.mojo":170:1)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mojo":239:5)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mojo":242:9)
// CHECK-DAG: #[[LOC4:.*]] = loc("foo.mojo":1473:5)
#loc1 = loc("foo.mojo":170:1)
#loc2 = loc("foo.mojo":239:5)
#loc3 = loc("foo.mojo":242:9)
#loc4 = loc("foo.mojo":1473:5)
#loc5 = loc("foo.mojo":1489:9)

// CHECK-DAG: #[[SP_FOO:.*]] = #debuginfo.subprogram<name = <"foo">
// CHECK-DAG: #[[SP_CLOSURE:.*]] = #debuginfo.subprogram<name = <"Closure">
// CHECK-DAG: #[[SP_NESTED:.*]] = #debuginfo.subprogram<name = <"NestedClosure">
#sp = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spClosure = #debuginfo.subprogram<name = <"Closure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spNested = #debuginfo.subprogram<name = <"NestedClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spCap = #debuginfo.subprogram<name = <"Capturing">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spNestedCap = #debuginfo.subprogram<name = <"NestedCapturing">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

// CHECK-DAG: #[[LOC_NESTED]] = loc(fused<#[[SP_NESTED]]>[#[[LOC3]]])
// CHECK-DAG: #[[LOC_CLOSURE]] = loc(fused<#[[SP_CLOSURE]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_NESTED_DEC]] = loc(fused<#[[SP_CLOSURE]]>[#[[LOC3]]])
// CHECK-DAG: #[[LOC_FOO]] = loc(fused<#[[SP_FOO]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CLOSURE_DEC]] = loc(fused<#[[SP_FOO]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_CAP]] = loc(fused<#[[SP_FOO]]>[#[[LOC4]]])
#locFoo = loc(fused<#sp>[#loc1])
#locClosure = loc(fused<#spClosure>[#loc2])
#locNested = loc(fused<#spNested>[#loc3])
#locCap = loc(fused<#spCap>[#loc4])
#locNestedCap = loc(fused<#spNestedCap>[#loc5])

// -----

// COM: Use of 'a' appears only in a location inside the closure.

// CHECK-LABEL: @outline_closures_ref_closure<a>()
// CHECK-NEXT: kgen.return loc([[LOC:#.*]])

// CHECK-LABEL: @outline_closures_ref<a>
kgen.generator @outline_closures_ref<a>() {
  // CHECK-NEXT: declare closure: () -> () = <@outline_closures_ref_closure<a>>
  kgen.param.declare.region closure = () {
    kgen.return loc(fused<#kgen.param.decl.ref<"a"> : index>["a:0:0"])
  }
  kgen.return
}

// CHECK: [[LOC]] = loc(fused<#kgen.param.decl.ref<"a"> : index>[

// -----

// COM: Fix for MOCO-869:
// COM: decl is defined in the nested scope of `kgen.param.for` which is not at or above current
// COM: `param.declare.region`. It is safe to ignore and should not crash the compiler.
// CHECK-LABEL: @ignore_param_defined_in_nested_non_decl_region_scope()
kgen.generator @ignore_param_defined_in_nested_non_decl_region_scope() {
  kgen.param.declare.region closure = () {
    kgen.param.for decl: index in :index 2 iter :(index) -> !kgen.none @wrapper {
      kgen.param.for.continue loc(fused<#kgen.param.decl.ref<"decl"> : index>["x:0"])
    } else {
      kgen.param.yield
    }
    kgen.return
  }
  kgen.return
}

// CHECK: #[[LOC_X:.*]] = loc("x:0")
// CHECK: #[[LOC_CONTINUE:.*]] = loc(fused<#kgen.param.decl.ref<"decl"> : index>[#[[LOC_X]]])
