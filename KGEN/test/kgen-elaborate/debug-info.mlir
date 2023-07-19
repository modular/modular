// RUN: kgen-elaborate-opt %s -elaborate-generators=elaborate-locations=true -mlir-print-debuginfo | FileCheck %s

// Check that debug info gets resolved during elaboration.

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>

// CHECK-DAG: #[[LOC_TRY_FILE:.*]] = loc("silly.mlir":17:3)
#locTry = loc("silly.mlir":17:3)

// CHECK-DAG: ![[PARAM_TYPE:.*]] = !debuginfo.unresolved<index>
!unresolved = !debuginfo.unresolved<!kgen.paramref<ty>>

// CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<() -> (index): DW_CC_normal>
!subroutine = !debuginfo.subroutine<() -> (!unresolved): DW_CC_normal>

// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}} name = "takeFnContextualType", linkageName = "takeFnContextualType,ty=index,fn=sillyFn", {{.*}}> : ![[SP_TYPE]]
// CHECK-DAG: #[[LOC_TRY:.*]] = loc(fused<#[[SP]]>[#[[LOC_TRY_FILE]]])
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "takeFnContextualType", linkageName = "takeFnContextualType", file = #file, line = 2, scopeLine = 2, subprogramFlags = "Definition|Optimized"> : !subroutine

// CHECK: #[[VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], {{.*}}> : ![[PARAM_TYPE]]
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "0", file = #file, line = 3, arg = 0, alignInBits = 0> : !unresolved

// CHECK-LABEL: kgen.func @"takeFnContextualType,ty=index,fn=sillyFn"() -> index
// CHECK:   %[[RES:.*]] = kgen.call @sillyFn() : () -> index loc(#[[CALL_LOC:.*]])
// CHECK:   debuginfo.value #[[VAR]] = %[[RES]] : index loc(#[[CALL_LOC:.*]])
// CHECK:   kgen.param.constant = <17> loc(#[[FW_LOC:.*]])
// CHECK:   lit.try {
// CHECK:   } except (%arg0: index loc(fused<#[[SP]]>[#[[LOC_TRY_FILE]]])) {
// CHECK:     lit.try.yield loc(#[[LOC_TRY]])
// CHECK:   kgen.return %[[RES]] : index loc(#[[RET_LOC:.*]])
// CHECK: } loc(#[[SP_LOC:.*]])
kgen.generator @takeFnContextualType<ty: type, fn: () -> !kgen.paramref<ty>>() -> !kgen.paramref<ty> {
  %0 = kgen.call_param[() -> !kgen.paramref<ty>: fn]() loc(#loc11)
  debuginfo.value #local_variable = %0 : !kgen.paramref<ty> loc(#loc11)
  %1 = kgen.param.constant = <17> loc(#locFwParam)
  kgen.param.declare a = <1> loc(#loc11)
  lit.try {
    lit.try.yield loc(#loc11)
  } except (%arg0: index loc(fused<#subprogram>[#locTry])) {
    lit.try.yield loc(fused<#subprogram>[#locTry])
  } else {
    lit.try.yield loc(#loc11)
  } loc(#loc11)
  kgen.return %0 : !kgen.paramref<ty> loc(#locRet)
} loc(#loc10)

kgen.generator @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0 : index
}

kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<:type index, :() -> index @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-DAG: #[[FILE_LOC1:.*]] = loc("test.mlir":2:3)
// CHECK-DAG: #[[SP_LOC]] = loc(fused<#[[SP]]>[#[[FILE_LOC1]]])

// CHECK-DAG: #[[FILE_LOC2:.*]] = loc("test.mlir":3:10)
// CHECK-DAG: #[[CALL_LOC]] = loc(fused<#[[SP]]>[#[[FILE_LOC2]]])

// CHECK-DAG: #[[FILE_LOC3:.*]] = loc("test.mlir":4:3)
// CHECK-DAG: #[[PARAM_REF_LOC:.*]] = loc(fused<1 : index>[#[[FILE_LOC3]]])
// CHECK-DAG: #[[FW_LOC]] = loc(fused<#[[SP]]>[#[[PARAM_REF_LOC]]])

// CHECK-DAG: #[[FILE_LOC4:.*]] = loc("test.mlir":5:5)
// CHECK-DAG: #[[RET_LOC]] = loc(fused<#[[SP]]>[#[[FILE_LOC4]]])

#loc10 = loc(fused<#subprogram>["test.mlir":2:3])
#loc11 = loc(fused<#subprogram>["test.mlir":3:10])
#paramRefLoc = loc(fused<#kgen.param.decl.ref<"a">>["test.mlir":4:3])
#locFwParam = loc(fused<#subprogram>[#paramRefLoc])
#locRet = loc(fused<#subprogram>["test.mlir":5:5])
