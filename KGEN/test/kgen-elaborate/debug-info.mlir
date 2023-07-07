// RUN: kgen-elaborate-opt %s -elaborate-generators -mlir-print-debuginfo | FileCheck %s

// Check that debug info gets resolved during elaboration.

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>

// CHECK-DAG: ![[PARAM_TYPE:.*]] = !debuginfo.unresolved<index>
!unresolved = !debuginfo.unresolved<!kgen.paramref<ty>>

// CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<() -> (index): DW_CC_normal>
!subroutine = !debuginfo.subroutine<() -> (!unresolved): DW_CC_normal>

// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}} name = "takeFnContextualType", linkageName = "takeFnContextualType,ty=index,fn=sillyFn", {{.*}}> : ![[SP_TYPE]]
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "takeFnContextualType", linkageName = "takeFnContextualType", file = #file, line = 2, scopeLine = 2, subprogramFlags = "Definition|Optimized"> : !subroutine

// CHECK: #[[VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], {{.*}}> : ![[PARAM_TYPE]]
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "0", file = #file, line = 3, arg = 0, alignInBits = 0> : !unresolved

// CHECK-LABEL: kgen.func @"takeFnContextualType,ty=index,fn=sillyFn"() -> index
// CHECK:   %[[RES:.*]] = kgen.call @sillyFn() : () -> index loc(#[[CALL_LOC:.*]])
// CHECK:   debuginfo.value #[[VAR]] = %[[RES]] : index
// CHECK:   kgen.return %[[RES]] : index loc(#[[RET_LOC:.*]])
// CHECK: } loc(#[[SP_LOC:.*]])
kgen.generator @takeFnContextualType<ty: type, fn: () -> !kgen.paramref<ty>>() -> !kgen.paramref<ty> {
  %0 = kgen.call_param[() -> !kgen.paramref<ty>: fn]()  loc(#loc11)
  debuginfo.value #local_variable = %0 : !kgen.paramref<ty> loc(#loc11)
  kgen.return %0 : !kgen.paramref<ty> loc(#loc12)
} loc(#loc10)

kgen.generator @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0 : index
}

kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<:type index, :() -> index @sillyFn>() : () -> index
  kgen.return %0 : index
}

// COM: TODO: fix resolving parameters in debuginfo
// CHECK-DAG: ![[SP_TYPE1:.*]] = !debuginfo.subroutine<() -> (!kgen.paramref<ty>): DW_CC_normal>
// CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<{{.*}} name = "takeFnContextualType", linkageName = "takeFnContextualType,ty=index,fn=sillyFn", {{.*}}> : ![[SP_TYPE1]]

// CHECK-DAG: #[[FILE_LOC1:.*]] = loc("test.mlir":2:3)
// CHECK-DAG: #[[SP_LOC]] = loc(fused<#[[SP]]>[#[[FILE_LOC1]]])

// CHECK-DAG: #[[FILE_LOC2:.*]] = loc("test.mlir":3:10)
// CHECK-DAG: #[[CALL_LOC]] = loc(fused<#[[SP1]]>[#[[FILE_LOC2]]])

// CHECK-DAG: #[[FILE_LOC3:.*]] = loc("test.mlir":4:5)
// CHECK-DAG: #[[RET_LOC]] = loc(fused<#[[SP1]]>[#[[FILE_LOC3]]])

#loc10 = loc(fused<#subprogram>["test.mlir":2:3])
#loc11 = loc(fused<#subprogram>["test.mlir":3:10])
#loc12 = loc(fused<#subprogram>["test.mlir":4:5])
