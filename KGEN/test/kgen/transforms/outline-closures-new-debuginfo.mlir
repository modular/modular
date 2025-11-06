// RUN: kgen-opt %s -split-input-file -outline-closures-new=debug-build=true -mlir-print-debuginfo -verify-parameters | FileCheck %s

// COM: Use of 'C' appears only in a location inside the closure.

// Provide a struct generator for the escaping closure
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping> {
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> read_mem) -> () = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

// CHECK: kgen.generator @foo_fn<C>(%arg0: !kgen.pointer<none> loc({{.*}}) read_mem)
// CHECK-NEXT:   kgen.return loc(#loc
// CHECK-NEXT: } loc(#loc
kgen.generator @foo<C>() {
  %3 = kgen.closure.init()() -> () {
	  kgen.return loc(fused<#kgen.param.decl.ref<"C"> : index>["C:0:0"])
  } : (), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
  kgen.return
}

// -----

// COM: The parameter references are erased from the locations of del, move, and copy and replaced with empty function subroutine type
// TODO: recreate the actual signature in debug info if needed.

// CHECK: !debuginfo.subroutine<() -> (): DW_CC_normal>


!ti2Eptr2 = !debuginfo.ti.ptr<!kgen.param<U>>
#captureParams_name = #debuginfo.source_name<(fn)"captureParams" from <(module)"delete-me">>
#file = #debuginfo.file<"delete-me.mojo" in "">
#loc11 = loc("delete-me.mojo":6:42)
#loc12 = loc("delete-me.mojo":6:56)
#loc14 = loc("delete-me.mojo":7:28)
!subroutine1 = !debuginfo.subroutine<(!ti2Eptr2) -> (index): DW_CC_normal>
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full, nameTableKind = None>
#hasParams_name = #debuginfo.source_name<(fn)"hasParams" from #captureParams_name>
#subprogram1 = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = #hasParams_name, linkageName = "hasParams", file = #file, line = 7, scopeLine = 7, subprogramFlags = "Definition|Optimized"> : !subroutine1
module {
  kgen.struct.generator @"captureParams::hasParams"<CAPTURES: !kgen.param_closure<@captureParams "hasParams">> = !kgen.closure<@captureParams, "hasParams" nonescaping>{
    kgen.conformance @Copyable {
      kgen.witness "__copyinit__($0)" : (!kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>> read_mem, !kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@captureParams, "hasParams", #kgen.closure_method<copy>, <:!kgen.param_closure<@captureParams "hasParams"> CAPTURES>> loc(#loc)
    } loc(#loc)
    kgen.conformance @AnyType {
      kgen.witness "__del__($0$)" : (!kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@captureParams, "hasParams", #kgen.closure_method<del>, <:!kgen.param_closure<@captureParams "hasParams"> CAPTURES>> loc(#loc)
    } loc(#loc)
    kgen.conformance @Movable {
      kgen.witness "__moveinit__($0$)" : (!kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@captureParams, "hasParams", #kgen.closure_method<move>, <:!kgen.param_closure<@captureParams "hasParams"> CAPTURES>> loc(#loc)
    } loc(#loc)
    kgen.conformance @"fn[U: Trait](impl: U) -> Int" {
      kgen.witness "__call__[delete-me::Trait]($0,$1)" : <type>(!kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>> read_mem, !kgen.pointer<*(0,0)> read_mem) capturing -> index = #kgen.closure.symbol<@captureParams, "hasParams", #kgen.closure_method<call>, <:type ?, :!kgen.param_closure<@captureParams "hasParams"> CAPTURES>> loc(#loc)
    } loc(#loc)
  } loc(#loc)
  kgen.generator @captureParams<X: type, Y: type>(%arg0: !kgen.pointer<X> loc("delete-me.mojo":6:42) mut, %arg1: !kgen.pointer<Y> loc("delete-me.mojo":6:56) mut) -> !kgen.none attributes {sourceName = "captureParams"} {
    %0 = kgen.closure.init()<U: type>(%arg2: !kgen.pointer<U> loc("delete-me.mojo":7:28) read_mem) capturing -> index {
      %idx1 = index.constant 1 loc(#loc19)
      kgen.return %idx1 : index loc(#loc19)
    } : (), !kgen.pointer<!kgen.closure<@captureParams, "hasParams" nonescaping>>, #subprogram1 loc(#loc18)
    %none = kgen.param.constant: none = <#kgen.none> loc(#loc)
    kgen.return %none : !kgen.none loc(#loc)
  } loc(#loc)
} loc(#loc)
!ti2Eptr = !debuginfo.ti.ptr<!kgen.param<X>>
!ti2Eptr1 = !debuginfo.ti.ptr<!kgen.param<Y>>
#loc = loc("delete-me.mlir":0:0)
#loc13 = loc("delete-me.mojo":7:8)
#loc15 = loc("delete-me.mojo":8:9)
!subroutine = !debuginfo.subroutine<(!ti2Eptr, !ti2Eptr1) -> (!kgen.none): DW_CC_normal>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = #captureParams_name, linkageName = "delete-me::captureParams[delete-me::Trait,delete-me::Trait]($0&,$1&)", file = #file, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized"> : !subroutine
#loc18 = loc(fused<#subprogram>[#loc13])
#loc19 = loc(fused<#subprogram1>[#loc15])
