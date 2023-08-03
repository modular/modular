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

// CHECK-LABEL: kgen.generator @foo() -> !pop.array<0, i1> {
// CHECK-NEXT:    kgen.param.declare Closure: () -> !pop.array<0, i8> = <@foo_Closure> loc(#[[LOC_CLOSURE_DEC:loc[0-9]*]])
// CHECK-NEXT:    kgen.param.declare OtherClosure: () -> () = <@foo_OtherClosure> loc(#[[LOC_FOO:loc[0-9]*]])
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[LOC_FOO]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i1> loc(#[[LOC_FOO]])
// CHECK-NEXT:  } loc(#[[LOC_FOO]])

kgen.generator @foo() -> !pop.array<0, i1> {
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

  %array = kgen.param.constant: array<0, i1> = <[]> loc(#locFoo)
  kgen.return %array : !pop.array<0, i1> loc(#locFoo)
} loc(#locFoo)

// CHECK-DAG: #[[LOC1]] = loc("foo.mojo":170:1)
// CHECK-DAG: #[[LOC2:loc[0-9]*]] = loc("foo.mojo":239:5)
// CHECK-DAG: #[[LOC3:loc[0-9]*]] = loc("foo.mojo":242:9)
#loc1 = loc("foo.mojo":170:1)
#loc2 = loc("foo.mojo":239:5)
#loc3 = loc("foo.mojo":242:9)

#file = #debuginfo.file<"foo.mojo" in "/">
#compile_unit1 = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>

// CHECK-DAG: #[[SP_FOO:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE:file[0-9]*]], name = "foo", linkageName = "foo",
// CHECK-DAG: #[[SP_CLOSURE:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = "Closure", linkageName = "foo_Closure",
// CHECK-DAG: #[[SP_NESTED:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = "NestedClosure", linkageName = "foo_NestedClosure",
#sp = #debuginfo.subprogram<
  compileUnit = #compile_unit1,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 170,
  scopeLine = 170,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!pop.array<0, i1>): DW_CC_normal>
#spClosure = #debuginfo.subprogram<
  compileUnit = #compile_unit1,
  scope = #file,
  name = "Closure",
  linkageName = "Closure",
  file = #file,
  line = 239,
  scopeLine = 239,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!pop.array<0, i8>): DW_CC_normal>
#spNested = #debuginfo.subprogram<
  compileUnit = #compile_unit1,
  scope = #file,
  name = "NestedClosure",
  linkageName = "NestedClosure",
  file = #file,
  line = 242,
  scopeLine = 242,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!pop.array<0, i32>): DW_CC_normal>

// CHECK-DAG: #[[LOC_NESTED]] = loc(fused<#[[SP_NESTED]]>[#[[LOC3]]])
// CHECK-DAG: #[[LOC_CLOSURE]] = loc(fused<#[[SP_CLOSURE]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_NESTED_DEC]] = loc(fused<#[[SP_CLOSURE]]>[#[[LOC3]]])
// CHECK-DAG: #[[LOC_FOO]] = loc(fused<#[[SP_FOO]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CLOSURE_DEC]] = loc(fused<#[[SP_FOO]]>[#[[LOC2]]])

#locFoo = loc(fused<#sp>[#loc1])
#locClosure = loc(fused<#spClosure>[#loc2])
#locNested = loc(fused<#spNested>[#loc3])
