// RUN: kgen-opt -force-inline=update-debug-info=true -mlir-print-debuginfo %s | FileCheck %s

// COM: The attributes may be printed before or after the functions under test,
// COM: so we try to keep the attribute close to its corresponding CHECK
// COM: statement, if possible. Moreover, to avoid some issues with FileCheck
// COM: getting confused with CHECK-DAG statements (and to reduce duplicate test
// COM: code), we also don't use -split-file.

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
#callerSp = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "caller",
  linkageName = "caller",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>
// CHECK-DAG: #[[SP_ASYNC:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "call_async", linkageName = "call_async", file = #file, line = 50, scopeLine = 50,
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

// CHECK-DAG: #[[LOC_ASYNC_CALLER:.*]] = loc("bar.mlir":18:7)
// CHECK-DAG: #[[LOC_SCOPED_CALLER:.*]] = loc(fused<#[[SP_ASYNC]]>[#[[LOC_ASYNC_CALLER]]])
#locAsyncCaller = loc(fused<#asyncCallerSp>["bar.mlir":18:7])

// CHECK-DAG: #[[LOC_CALLSITE_FILE:.*]] = loc("bar.mlir":27:8)
#locCallsite = loc("bar.mlir":27:8)

// CHECK-DAG: #[[LOC_CALLSITE:.*]] = loc(fused<{{.*}}#[[LOC_CALLSITE_FILE]]
#locCaller = loc(fused<#callerSp>[#locCallsite])
// CHECK-DAG: #[[INLINED_LOC:.*]] = loc(callsite(#[[LOC_SCOPED_CALLER]] at

// -------------------------------------------------------------------------- //
// Test nodebug behavior for debuginfo.value ops.
// -------------------------------------------------------------------------- //

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

// -------------------------------------------------------------------------- //
// Test location handling of inlining.
// -------------------------------------------------------------------------- //

#loc = loc("foo.mlir":13:1)
#locArg = loc("foo.mlir":13:12)
#locCallee = loc(fused<#calleeSp>[#loc])

kgen.func @inline_me(%arg0: index) -> index always_inline {
  debuginfo.value #local_variable = %arg0 : index loc(fused<#calleeSp>[#locArg])
  kgen.return %arg0: index loc(#locCallee)
} loc(#locCallee)

// CHECK-LABEL: kgen.func @call_inline_me
kgen.func @call_inline_me() -> index {
  %0 = index.constant 3 loc(#locCaller)
  // CHECK: %idx3 = index.constant 3
  // CHECK-NEXT: debuginfo.value #local_variable = %idx3 : index loc(#[[LOC_VALUE_INLINED:.*]])
  %1 = kgen.call @inline_me(%0) : (index) -> index loc(#locCaller)
  kgen.return %1 : index loc(#locCaller)
} loc(#locCaller)

// -------------------------------------------------------------------------- //
// Test location handling of async closure staging.
// -------------------------------------------------------------------------- //

// CHECK-LABEL: kgen.func @call_async
kgen.func @call_async() -> !pop.coroutine<() -> (index)> {
  // CHECK-NEXT: %idx2 = index.constant 2 loc(#[[LOC_SCOPED_CALLER]])
  %idx2 = index.constant 2 loc(#locAsyncCaller)
  // CHECK-NEXT: lit.async.execute <() -> index>
  // CHECK-NEXT:   debuginfo.value #local_variable = %idx2 : index loc(#[[LOC_VALUE:.*]])
  // CHECK-NEXT:   lit.async.return %idx2 : index loc(#[[LOC_ASYNC_EXECUTE:.*]])
  // CHECK-NEXT: } {inliner_debuginfo_update = 1 : i8} callLoc(#[[LOC_SCOPED_CALLER]]) loc(#[[LOC_ASYNC_EXECUTE]])
  %coroHdl = lit.async.call[(index) async -> index: @inline_me](%idx2) loc(#locAsyncCaller)
  // CHECK-NEXT: kgen.return
  kgen.return %coroHdl : !pop.coroutine<() -> (index)> loc(#locAsyncCaller)
// CHECK-NEXT: } loc(#[[LOC_SCOPED_CALLER]])
} loc(#locAsyncCaller)

// -------------------------------------------------------------------------- //
// Test location handling of inlined closure staging.
// -------------------------------------------------------------------------- //

kgen.func @async_wrapper() -> !pop.coroutine<() -> (index)> always_inline {
  %idx2 = index.constant 3 loc(#locAsyncCaller)
  %0 = lit.async.call[(index) async -> index: @inline_me](%idx2) loc(#locAsyncCaller)
  kgen.return %0 : !pop.coroutine<() -> (index)> loc(#locAsyncCaller)
} loc(#locAsyncCaller)

// CHECK-LABEL: kgen.func @call_async_indirect
kgen.func @call_async_indirect() -> !pop.coroutine<() -> (index)> {
  // CHECK-NEXT: %idx3 = index.constant 3 loc(#[[INLINED_LOC]])
  // CHECK-NEXT: lit.async.execute <() -> index>
  // CHECK-NEXT:   debuginfo.value #local_variable = %idx3 : index loc(#[[LOC_VALUE]])
  // CHECK-NEXT:   lit.async.return %idx3 : index loc(#[[LOC_ASYNC_EXECUTE]])
  // CHECK-NEXT: } {inliner_debuginfo_update = 1 : i8} callLoc(#[[INLINED_LOC]]) loc(#[[LOC_ASYNC_EXECUTE]])
  %1 = kgen.call @async_wrapper() : () -> !pop.coroutine<() -> (index)> loc(#locCaller)
  kgen.return %1 : !pop.coroutine<() -> (index)> loc(#locCaller)
} loc(#locCaller)

// CHECK-DAG: #[[LOC:loc[0-9]+]] = loc("foo.mlir":13:1)
// CHECK-DAG: #[[LOC_ARG:loc[0-9]+]] = loc("foo.mlir":13:12)

// CHECK-DAG: #[[LOC_VALUE_INLINED]] = loc(callsite(#[[LOC_VALUE:loc[0-9]+]] at #[[LOC_CALLSITE]]))
// CHECK-DAG: #[[LOC_VALUE]] = loc(fused<#[[SP]]>[#[[LOC_ARG]]])
// CHECK-DAG: #[[LOC_ASYNC_EXECUTE]] = loc(fused<#[[SP]]>[#[[LOC]]])
