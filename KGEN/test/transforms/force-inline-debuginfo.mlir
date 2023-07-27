// RUN: kgen-opt -force-inline=update-debug-info=true -mlir-print-debuginfo %s | FileCheck %s

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "inline_me", linkageName = "inline_me", file = #file, line = 10, scopeLine = 10,
#calleeSp = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "inline_me",
  linkageName = "inline_me",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>
#asyncCallerSp = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "call_async",
  linkageName = "call_async",
  file = #file,
  line = 50,
  scopeLine = 50,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (!debuginfo.unresolved<!pop.coroutine<() -> (index)>>): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #calleeSp,
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
#locCallee = loc(fused<#calleeSp>[#loc])
#locCallsite = loc("bar.mlir":27:8)
#locAsyncCaller = loc(fused<#asyncCallerSp>["bar.mlir":18:7])

kgen.func @inline_me(%arg0: index) -> index always_inline {
  debuginfo.value #local_variable = %arg0 : index loc(fused<#calleeSp>[#locArg])
  kgen.return %arg0: index loc(#locCallee)
} loc(#locCallee)

// CHECK-LABEL: kgen.func @call_inline_me
kgen.func @call_inline_me() -> index {
  %0 = index.constant 3
  // CHECK: %idx3 = index.constant 3
  // CHECK-NEXT: debuginfo.value #local_variable = %idx3 : index loc(#[[LOC_VALUE_INLINED:loc[0-9]+]])
  %1 = kgen.call @inline_me(%0) : (index) -> index loc(#locCallsite)
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @call_async
kgen.func @call_async() -> !pop.coroutine<() -> (index)> {
  // CHECK-NEXT: %idx2 = index.constant 2 loc(#[[LOC_SCOPED_CALLER:loc[0-9]+]])
  %idx2 = index.constant 2 loc(#locAsyncCaller)
  // CHECK-NEXT: %[[RES:.*]] = lit.async.execute <() -> index>
  // CHECK-NEXT:   debuginfo.value #local_variable = %idx2 : index loc(#[[LOC_VALUE:loc[0-9]+]])
  // CHECK-NEXT:   lit.async.return %idx2 : index loc(#[[LOC_ASYNC_EXECUTE:loc[0-9]+]])
  // CHECK-NEXT: } {inliner_debuginfo_update = 1 : i8} loc(#[[LOC_ASYNC_EXECUTE]])
  %coroHdl = lit.async.call[(index) async -> index: @inline_me](%idx2) loc(#locAsyncCaller)
  // CHECK-NEXT: kgen.return %[[RES]]
  kgen.return %coroHdl : !pop.coroutine<() -> (index)> loc(#locAsyncCaller)
// CHECK-NEXT: } loc(#[[LOC_SCOPED_CALLER]])
} loc(#locAsyncCaller)

// CHECK-DAG: #[[LOC:loc[0-9]+]] = loc("foo.mlir":13:1)
// CHECK-DAG: #[[LOC_ARG:loc[0-9]+]] = loc("foo.mlir":13:12)
// CHECK-DAG: #[[LOC_CALLSITE:loc[0-9]+]] = loc("bar.mlir":27:8)
// CHECK-DAG: #[[LOC_ASYNC_CALLER:loc[0-9]+]] = loc("bar.mlir":18:7)

// CHECK-DAG: #[[SP_ASYNC:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "call_async", linkageName = "call_async", file = #file, line = 50, scopeLine = 50,

// CHECK-DAG: #[[LOC_VALUE_INLINED]] = loc(callsite(#[[LOC_VALUE:loc[0-9]+]] at #[[LOC_CALLSITE]]))
// CHECK-DAG: #[[LOC_VALUE]] = loc(fused<#[[SP]]>[#[[LOC_ARG]]])
// CHECK-DAG: #[[LOC_SCOPED_CALLER]] = loc(fused<#[[SP_ASYNC]]>[#[[LOC_ASYNC_CALLER]]])
// CHECK-DAG: #[[LOC_ASYNC_EXECUTE]] = loc(fused<#[[SP]]>[#[[LOC]]])
