// RUN: kgen-opt -force-inline=update-debug-info=true -mlir-print-debuginfo %s | FileCheck %s

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
> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>
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

// CHECK-LABEL: kgen.func @call_nodebug_inline_me
kgen.func @call_nodebug_inline_me() -> index {
  %0 = index.constant 3
  // CHECK: index.add %idx3, %idx3
  // CHECK-NOT: debuginfo.value
  %1 = kgen.call @nodebug_inline_me(%0) : (index) -> index
  kgen.return %1 : index
}

#loc = loc("foo.mlir":13:1)
#locArg = loc("foo.mlir":13:12)
#locCallee = loc(fused<#subprogram>[#loc])
#locCallsite = loc("bar.mlir":27:8)

kgen.func @inline_me(%arg0: index) -> index always_inline {
  debuginfo.value #local_variable = %arg0 : index loc(fused<#subprogram>[#locArg])
  kgen.return %arg0: index loc(#locCallee)
} loc(#locCallee)

// CHECK-LABEL: kgen.func @call_inline_me
kgen.func @call_inline_me() -> index {
  %0 = index.constant 3
  // CHECK: %idx3 = index.constant 3
  // CHECK-NEXT: debuginfo.value #local_variable = %idx3 : index loc(#[[LOC_VALUE:loc[0-9]+]])
  %1 = kgen.call @inline_me(%0) : (index) -> index loc(#locCallsite)
  kgen.return %1 : index
}

// CHECK-DAG: #[[LOC_VALUE]] = loc(callsite(#[[LOC_CALLEE:loc[0-9]+]] at #[[LOC_CALLSITE:loc[0-9]+]]))
// CHECK-DAG: #[[LOC_CALLEE]] = loc(fused<#subprogram>[#[[LOC:loc[0-9]+]]])
// CHECK-DAG: #[[LOC]] = loc("foo.mlir":13:12)
// CHECK-DAG: #[[LOC_CALLSITE]] = loc("bar.mlir":27:8)
