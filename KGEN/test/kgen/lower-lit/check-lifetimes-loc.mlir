// RUN: kgen-opt %s -mlir-print-debuginfo -check-lifetimes | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"__del__">> : !debuginfo.subroutine<(!kgen.pointer<@HasMemFields>) -> (!kgen.none): DW_CC_normal>

// CHECK-DAG: #[[FILE_LOC:.*]] = loc("foo.mlir":
// CHECK-DAG: #[[LOC:.*]] = loc(fused<#subprogram>[#[[FILE_LOC]]])
#loc = loc("foo.mlir":43:25)
#loc1 = loc(fused<#subprogram>[#loc])

lit.struct.decl @S attributes {destructor = #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : index
}

lit.struct.decl @HasMemFields attributes {destructor = #kgen.symbol.constant<@HasMemFields::@__del__> : !lit.signature<[1](!lit.ref<@HasMemFields, mut *[0,0]> owned_in_mem) -> !kgen.none>} {
  lit.struct.field a : !lit.struct<@S>

  lit.func @__del__[mut dtorlife](%self: !lit.ref<@HasMemFields, mut dtorlife> loc(#loc) owned_in_mem) -> !kgen.none {
    // CHECK-DAG: [[VAR0:%.*]] = lit.ref.struct.ger %self[a]
    // CHECK-DAG: [[VAR1:%.*]] = lit.call @S::@__del__[mut dtorlife]([[VAR0]])
    lit.ownership.mark_destroyed %self : !lit.ref<@HasMemFields, mut dtorlife> loc(#loc1)
    %none = kgen.param.constant: none = <#kgen.none> loc(#loc1)
    // CHECK-DAG: kgen.return %{{.*}} : !kgen.none loc(#[[LOC]])
    kgen.return %none : !kgen.none loc(#loc1)
  } loc(#loc1)
}
