// RUN: kgen-opt -mlir-print-debuginfo -lower-closures %s | FileCheck %s

#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 44,
  scopeLine = 44,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "SomeClosure",
  linkageName = "SomeClosure",
  file = #file,
  line = 325,
  scopeLine = 325,
  subprogramFlags = "Definition|Optimized"
>  : !debuginfo.subroutine<() -> (!pop.array<0, i1>): DW_CC_normal>
#subprogram2 = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "OtherClosure",
  linkageName = "OtherClosure",
  file = #file,
  line = 412,
  scopeLine = 412,
  subprogramFlags = "Definition|Optimized"
>  : !debuginfo.subroutine<() -> (!pop.array<0, i1>): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":46:8)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc("bar.mlir":415:15)
#loc5 = loc(fused<#subprogram>[#loc1])
#loc6 = loc(fused<#subprogram>[#loc2])
#loc7 = loc(fused<#subprogram1>[#loc3])
#loc8 = loc(fused<#subprogram2>[#loc4])

// CHECK-LABEL: kgen.func @foo_async_closure()
// CHECK-NEXT:    %0 = pop.coroutine.handle : <() -> !pop.array<0, i1>> loc(#[[FOO_ASYNC_CL_LOC:.*]])
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[FOO_ASYNC_CL_CONST_LOC:.*]])
// CHECK:         kgen.return %0 : !pop.coroutine<() -> !pop.array<0, i1>> loc(#[[FOO_ASYNC_CL_LOC]])
// CHECK-NEXT:  } loc(#[[FOO_ASYNC_CL_LOC]])

// CHECK-LABEL: kgen.func @foo_closure()
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[FOO_CL_CONST_LOC:.*]])
// CHECK-NEXT:    kgen.param.constant: array<2, i1> = <[1, 1]> loc(#[[FOO_CL_LOC:.*]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i1> loc(#[[FOO_CL_LOC]])
// CHECK-NEXT:  } loc(#[[FOO_CL_LOC]])

// CHECK-LABEL: kgen.func @foo()
kgen.func @foo() {
  // CHECK-NEXT: kgen.param.constant: array<0, i1> = <[]> loc(#[[LOC_CALLSITE:.*]])
  %array = kgen.param.constant: array<0, i1> = <[]> loc(#loc6)

  // CHECK-NEXT: kgen.call @foo_async_closure() : () -> !pop.coroutine<() -> !pop.array<0, i1>> loc(#[[LOC_CALLSITE]])
  %0 = lit.async.execute <() -> !pop.array<0, i1>> {
    %array_1 = kgen.param.constant: array<1, i1> = <[1]> loc(#loc7)
    lit.async.return %array : !pop.array<0, i1> loc(#loc7)
  } {inliner_debuginfo_update = 1 : i8} callLoc(#loc6) loc(#loc7)

  // CHECK-NEXT: kgen.create_closure [() capturing -> !pop.array<0, i1>: @foo_closure]()  loc(#[[LOC_CALLSITE]])
  %1 = kgen.stage_closure = () capturing -> !pop.array<0, i1> {
    %array_1 = kgen.param.constant: array<2, i1> = <[1, 1]> loc(#loc8)
    kgen.return %array : !pop.array<0, i1> loc(#loc8)
  } callLoc(#loc6) loc(#loc8)

  // CHECK-NEXT: kgen.return
  kgen.return loc(#loc5)
} loc(#loc5)

// CHECK-DAG: #[[SP_ASYNC_CL:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "foo_async_closure", linkageName = "foo_async_closure", file = #file, line = 325, scopeLine = 325,
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "foo", linkageName = "foo", file = #file, line = 44, scopeLine = 44,
// CHECK-DAG: #[[SP_CL:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "foo_closure", linkageName = "foo_closure", file = #file, line = 412, scopeLine = 412,

// CHECK-DAG: #[[SOME_CL_LOC:.*]] = loc("bar.mlir":327:17)
// CHECK-DAG: #[[FOO_ASYNC_CL_LOC]] = loc(fused<#[[SP_ASYNC_CL]]>[#[[SOME_CL_LOC]]])
// CHECK-DAG: #[[FOO_ASYNC_CL_CONST_LOC]] = loc(callsite(#[[LOC_CALLSITE]] at #[[FOO_ASYNC_CL_LOC]]))

// CHECK-DAG: #[[OTHER_CL_LOC:.*]] = loc("bar.mlir":415:15)
// CHECK-DAG: #[[FOO_CL_LOC]] = loc(fused<#[[SP_CL]]>[#[[OTHER_CL_LOC]]])
// CHECK-DAG: #[[FOO_CL_CONST_LOC]] = loc(callsite(#[[LOC_CALLSITE]] at #[[FOO_CL_LOC]]))
