// RUN: kgen-opt %s -mlir-print-debuginfo -check-lifetimes | FileCheck %s

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
    compileUnit = #compile_unit,
    scope = #file,
    name = "__del__",
    linkageName = "__del__(@HasMemFields::@__del__)",
    file = #file,
    line = 156,
    scopeLine = 156,
    subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<(!kgen.pointer<@HasMemFields>) -> (!lit.none): DW_CC_normal>

// CHECK-DAG: #[[FILE_LOC:.*]] = loc("foo.mlir":
// CHECK-DAG: #[[LOC:.*]] = loc(fused<#subprogram>[#[[FILE_LOC]]])
#loc = loc("foo.mlir":43:25)
#loc1 = loc(fused<#subprogram>[#loc])

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !kgen.signature<(!kgen.pointer<@S> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : index
}

lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !kgen.signature<(!kgen.pointer<@HasMemFields> owned_in_mem) -> !lit.none>} {
  lit.struct.field a : !kgen.declref<@S>

  lit.func @__del__(%self: !kgen.pointer<@HasMemFields> loc(#loc) owned_in_mem) -> !lit.none {
    // CHECK-DAG: %[[VAR0:.*]] = lit.struct.gep %self[a] : <@S> from <@HasMemFields> loc(#[[LOC]])
    // CHECK-DAG: %[[VAR1:.*]] = kgen.call @S::@__del__(%[[VAR0]]) : (!kgen.pointer<@S> owned_in_mem) -> !lit.none loc(#[[LOC]])
    lit.ownership.mark_destroyed %self : !kgen.pointer<@HasMemFields> loc(#loc1)
    %none = kgen.param.constant: !lit.none = <#lit.none> loc(#loc1)
    // CHECK-DAG: kgen.return %{{.*}} : !lit.none loc(#[[LOC]])
    kgen.return %none : !lit.none loc(#loc1)
  } loc(#loc1)
}
