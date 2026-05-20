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

// CHECK: kgen.generator @foo_fn<C>(%arg0: !kgen.pointer<struct<() memoryOnly>> loc({{.*}}) read_mem)
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

// -----

// COM: Ensure that ops, debuginfo.value in this case, are not unconditionally wrapped in a subprogram scoped FusedLoc

// DebugInfo values have a location and a variable attribute
// The variable attribute contains a scope
// The location scope of the debuginfo.value must be a child of the variable's location scope
// if both are wrapped in a subprogram fused location, they will appear to have a sibling relationship
// rather than a child-parent, failing verification.
!subroutine1 = !debuginfo.subroutine<(index) -> (index): DW_CC_normal>
!unresolved = !debuginfo.unresolved<index>
#expr2Eirvalue = #debuginfo.expr.irvalue : !kgen.pointer<index>
#file = #debuginfo.file<"delete-me.mojo" in "">
#loc2 = loc("delete-me.mlir":32:21)
#loc4 = loc("delete-me.mlir":33:35)
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full, nameTableKind = None>
#expr2Ederef = #debuginfo.expr.deref<#expr2Eirvalue> : index
#subprogram1 = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <(fn)"closure">, linkageName = "closure", file = #file, line = 13, scopeLine = 13, subprogramFlags = "Definition|Optimized"> : !subroutine1
#lexical_block1 = #debuginfo.lexical_block<scope = #subprogram1, file = #file, line = 16, column = 17>
#local_variable = #debuginfo.local_variable<scope = #lexical_block1, name = "j", file = #file, line = 15, flags = Zero> : !unresolved
module {
  kgen.struct.generator @"foo::closure"<CAPTURES: !kgen.param_closure<@foo "closure">> = !kgen.closure<@foo, "closure" nonescaping>{
    kgen.conformance @"fn(x: Int) -> Int" {
      kgen.witness "__call__($0,::Int)" : (!kgen.pointer<!kgen.closure<@foo, "closure" nonescaping>> read_mem, index) capturing -> index = #kgen.closure.symbol<@foo, "closure", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "closure"> CAPTURES>> loc(#loc11)
    } attributes {traitRef = @"fn(x: Int) -> Int"} loc(#loc11)
  } loc(#loc11)
  kgen.generator @foo(%arg0: index loc("delete-me.mlir":32:21)) {
    %0 = kgen.closure.init(%arg0)(%arg1: index loc("delete-me.mlir":33:35)) capturing -> index always_inline {
      hlcf.loop "__loop_1" {
        %1 = pop.stack_allocation 1 x index marked loc(#loc12)
        // CHECK:  debuginfo.value #local_variable #expr2Ederef = %2 : !kgen.pointer<index> loc([[LOC:#loc.*]])
        debuginfo.value #local_variable #expr2Ederef = %1 : !kgen.pointer<index> loc(#loc13)
        hlcf.continue loc(#loc12)
      } loc(#loc10)
      kgen.return %arg0 : index loc(#loc10)
    } : (index), !kgen.pointer<!kgen.closure<@foo, "closure" nonescaping>>, #subprogram1 loc(#loc9)
    kgen.unreachable loc(#loc9)
  } loc(#loc8)
} loc(#loc)
// CHECK-DAG: [[LOC]] = loc(fused<#lexical_block{{.*}}>[#loc
!subroutine = !debuginfo.subroutine<(index) -> (!kgen.none): DW_CC_normal>
#loc = loc("delete-me.mlir":0:0)
#loc1 = loc("delete-me.mojo":9:4)
#loc3 = loc("delete-me.mojo":13:12)
#loc5 = loc("delete-me.mojo":15:13)
#loc6 = loc("delete-me.mojo":15:17)
#loc7 = loc("delete-me.mojo":15:27)
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <(fn)"callIt">, linkageName = "callIt", file = #file, line = 9, scopeLine = 9, subprogramFlags = "Definition|Optimized"> : !subroutine
#lexical_block = #debuginfo.lexical_block<scope = #subprogram, file = #file, line = 12, column = 9>
#loc8 = loc(fused<#subprogram>[#loc1])
#loc9 = loc(fused<#subprogram>[#loc3])
#loc10 = loc(fused<#subprogram1>[#loc5])
#loc11 = loc(fused<#lexical_block>[#loc1])
#loc12 = loc(fused<#lexical_block1>[#loc6])
#loc13 = loc(fused<#lexical_block1>[#loc7])

// -----

// COM: Use of 'C' appears only in the kgen.closure.init op's own location,
// COM: not in any body op. This tests the region.getParentOp()->getLoc()
// COM: scanning path added to collectCapturedParams.

// Provide a struct generator for the escaping closure.
kgen.struct.generator @"bar::fn"<CAPTURES: !kgen.param_closure<@"bar" "fn">> = !kgen.closure<@"bar", "fn" escaping> {
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@bar, "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@bar, "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"bar", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"bar" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@bar, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"bar", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"bar" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@bar, "fn" escaping>> read_mem) -> () = #kgen.closure.symbol<@"bar", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"bar" "fn"> CAPTURES>>
  }
}

// CHECK: kgen.generator @bar_fn<C>(%arg0: !kgen.pointer<struct<() memoryOnly>> loc({{.*}}) read_mem)
kgen.generator @bar<C>() {
  %3 = kgen.closure.init()() -> () {
    kgen.return
  } : (), !kgen.pointer<!kgen.closure<@bar, "fn" escaping>> loc(fused<#kgen.param.decl.ref<"C"> : index>["bar.mlir":1:1])
  kgen.return
}

// -----

// COM: Use of 'D' appears only in the kgen.closure.init subprogram scope's
// COM: subroutine arg type, not in the op's own loc or body ops.
// COM: This tests the getSubprogramScope() scanning path added to
// COM: collectCapturedParams.

!ti_D = !debuginfo.ti.ptr<!kgen.param<D>>
!subroutine_D = !debuginfo.subroutine<(!ti_D) -> (): DW_CC_normal>
#baz_file = #debuginfo.file<"baz.mojo" in "">
#baz_compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #baz_file, producer = "Mojo", isOptimized = true, emissionKind = Full, nameTableKind = None>
#subprogram_D = #debuginfo.subprogram<compileUnit = #baz_compile_unit, scope = #baz_file, sourceName = <(fn)"baz_closure">, linkageName = "baz_closure", file = #baz_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized"> : !subroutine_D

kgen.struct.generator @"baz::fn"<CAPTURES: !kgen.param_closure<@"baz" "fn">> = !kgen.closure<@"baz", "fn" escaping> {
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@baz, "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@baz, "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"baz", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"baz" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@baz, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"baz", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"baz" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@baz, "fn" escaping>> read_mem) -> () = #kgen.closure.symbol<@"baz", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"baz" "fn"> CAPTURES>>
  }
}

// CHECK: kgen.generator @baz_fn<D: type>(%arg0: !kgen.pointer<struct<() memoryOnly>> loc({{.*}}) read_mem)
kgen.generator @baz<D: type>() {
  %3 = kgen.closure.init()() -> () {
    kgen.return
  } : (), !kgen.pointer<!kgen.closure<@baz, "fn" escaping>>, #subprogram_D loc("baz.mojo":0:0)
  kgen.return
}
