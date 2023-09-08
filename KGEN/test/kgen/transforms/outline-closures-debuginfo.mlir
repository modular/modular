// RUN: kgen-opt %s -outline-closures -mlir-print-debuginfo | FileCheck %s


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

// CHECK-LABEL: kgen.generator @foo(
// CHECK-SAME:      %[[ARG:.*]]: index
// CHECK-NEXT:    %[[VAL:.*]] = pop.struct.create(%[[ARG]]) : !pop.struct<index> loc(#[[LOC_FOO:.*]])
// CHECK-NEXT:    pop.compiler.global_store "foo_context_var_0", %[[VAL]] : !pop.struct<index> loc(#[[LOC_FOO]])
// CHECK-NEXT:    kgen.param.declare Closure: () -> !pop.array<0, i8> = <@foo_Closure> loc(#[[LOC_CLOSURE_DEC:.*]])
// CHECK-NEXT:    kgen.param.declare OtherClosure: () -> () = <@foo_OtherClosure> loc(#[[LOC_FOO]])
// CHECK-NEXT:    kgen.param.declare Capturing: <>() capturing -> () = <@foo_Capturing> loc(#[[LOC_CAP:.*]])
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

#file = #debuginfo.file<"foo.mojo" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>

// CHECK-DAG: #[[SP_FOO:.*]] = #debuginfo.subprogram<{{.*}}, name = "foo", linkageName = "foo",
// CHECK-DAG: #[[SP_CLOSURE:.*]] = #debuginfo.subprogram<{{.*}}, name = "Closure", linkageName = "foo_Closure",
// CHECK-DAG: #[[SP_NESTED:.*]] = #debuginfo.subprogram<{{.*}}, name = "NestedClosure", linkageName = "foo_NestedClosure",
#sp = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "foo", linkageName = "foo", file = #file, line = 170, scopeLine = 170, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!pop.array<0, i1>): DW_CC_normal>
#spClosure = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "Closure", linkageName = "Closure", file = #file, line = 239, scopeLine = 239, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!pop.array<0, i8>): DW_CC_normal>
#spNested = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "NestedClosure", linkageName = "NestedClosure", file = #file, line = 242, scopeLine = 242, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!pop.array<0, i32>): DW_CC_normal>
#spCap = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "Capturing", linkageName = "Capturing", file = #file, line = 1473, scopeLine = 1473, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spNestedCap = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "NestedCapturing", linkageName = "NestedCapturing", file = #file, line = 1489, scopeLine = 1489, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (index): DW_CC_normal>

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
