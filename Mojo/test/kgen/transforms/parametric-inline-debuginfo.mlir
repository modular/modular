// RUN: kgen-opt -mlir-print-debuginfo -inline-param=optimization-level=3 -verify-parameters -split-input-file %s | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"parent">> : !debuginfo.subroutine<() -> (index): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"callee">> : !debuginfo.subroutine<() -> (index): DW_CC_normal>
#loc = loc("foo.mlir":1:1)
#loc1 = loc("foo.mlir":10:10)
#loc2 = loc("foo.mlir":20:20)
#loc3 = loc("foo.mlir":30:30)
#loc4 = loc("foo.mlir":40:40)
#loc5 = loc("foo.mlir":50:50)
#loc8 = loc("foo.mlir":80:80)
#parentLoc = loc(fused<#subprogram>[#loc])
#callOpLoc = loc(fused<#subprogram>[#loc1])
#parentRetLoc = loc(fused<#subprogram>[#loc2])
#calleeLoc = loc(fused<#subprogram1>[#loc3])
#ret0 = loc(fused<#subprogram1>[#loc4])
#ret1 = loc(fused<#subprogram1>[#loc5])
#calleeMisc = loc(fused<#subprogram1>[#loc8])

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() -> index {
  // CHECK: %[[RES:.*]] = hlcf.loop "[[LABEL:.*]]" () -> index
    // CHECK-NEXT: index.constant 0 loc(#[[CONST_LOC:.*]])
    // CHECK: hlcf.if
      // CHECK-NEXT: hlcf.break "[[LABEL]]" %idx0 : index loc(#[[BREAK_LOC0:.*]])
    // CHECK: hlcf.break "[[LABEL]]" %idx0 : index loc(#[[BREAK_LOC1:.*]])
  // CHECK-NEXT: } loc(#[[CALL_LOC:.*]])
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index loc(#callOpLoc)
  // CHECK: return %[[RES]]
  hlcf.return %0 : index loc(#parentRetLoc)
} loc(#parentLoc)

// CHECK: kgen.generator @callee
kgen.generator @callee() -> index always_inline {
  // CHECK: index.constant 0 loc(#[[CALLEE_LOC:.*]])
  %0 = index.constant 0 loc(#calleeMisc)
  %false = kgen.param.constant: scalar<bool> = <false> loc(#calleeMisc)
  hlcf.if %false {
    // CHECK: hlcf.return %idx0 : index loc(#[[RET_LOC0:.*]])
    hlcf.return %0 : index loc(#ret0)
  } else {
    hlcf.yield loc(#calleeMisc)
  } loc(#calleeMisc)
  // CHECK: hlcf.return %idx0 : index loc(#[[RET_LOC1:.*]])
  hlcf.return %0 : index loc(#ret1)
} loc(#calleeLoc)

// CHECK: #[[CONST_LOC]] = loc("foo.mlir":80:80)
// CHECK: #[[BREAK_LOC0]] = loc(callsite(#[[RET_LOC0]] at #[[CALL_LOC]]))
// CHECK: #[[BREAK_LOC1]] = loc(callsite(#[[RET_LOC1]] at #[[CALL_LOC]]))

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<!kgen.param<T>>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!kgen.param<T>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<T: type>(%arg0: index) {
  // CHECK: kgen.param.declare T0: type = <index> loc(#[[CALL_LOC:.*]])
  // CHECK-NEXT: kgen.rebind %arg0 : index to !kgen.param<T0> loc(#[[CALL_LOC]])
  // CHECK-NEXT: hlcf.return
  kgen.call @nodebug_inline_me<:type index>(%arg0) : (index) -> () loc(#loc)
  hlcf.return loc(#loc)
} loc(#loc)

// CHECK-LABEL: kgen.generator @nodebug_inline_me
kgen.generator @nodebug_inline_me<T: type>(%arg0: !kgen.param<T>) always_inline_no_debug {
  hlcf.return loc(#loc)
} loc(#loc)

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"foo">, linkageName = "foo", file = #file, line = 10, scopeLine = 10, subprogramFlags = Definition> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.generator @no_debuginfo() -> index always_inline {
  %idx0 = index.constant 0
  hlcf.return %idx0 : index
}

// CHECK-LABEL: kgen.generator @has_debuginfo
kgen.generator @has_debuginfo() {
  // CHECK: index.constant 0 loc([[LOC:#.*]])
  kgen.call @no_debuginfo() : () -> index loc(#loc)
  hlcf.return loc(#loc)
} loc(#loc)

// CHECK: [[LOC]] = loc("{{.*}}":

