// RUN: kgen-opt -lower-calling-conventions %s -mlir-print-debuginfo -split-input-file | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (!kgen.none): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.func @main() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none> loc(#loc)
  kgen.return %none : !kgen.none loc(#loc)
} loc(#loc)

// CHECK: !debuginfo.subroutine<() -> ()

// -----

#subprogram = #debuginfo.subprogram<name = <"regtype_create_reg_stub">> : !debuginfo.subroutine<() -> (!kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>): DW_CC_normal>
#loc = loc(fused<#subprogram>["regtype_create_reg_stub.mlir":0:0])

// CHECK: #[[LOC_RAW:.+]] = loc("regtype_create_reg_stub.mlir":0:0)

kgen.func @regtype__moveinit__(%arg0: index owned) -> index {
  kgen.return %arg0 : index
}
// CHECK-LABEL: kgen.func @regtype_create_reg_stub
kgen.func @regtype_create_reg_stub() -> !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> {
  // CHECK: kgen.stage_closure
  // CHECK-NEXT: pop.pointer.bitcast {{.*}} loc(#[[LOC_RAW]])
  // CHECK: kgen.return loc(#[[LOC_RAW]])
  // CHECK-NEXT: } loc(#[[LOC_STAGECLOSURE:.+]])
  %0 = kgen.create_reg_stub [(index owned) -> index: @regtype__moveinit__] : <(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> loc(#loc)
  kgen.return %0 : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> loc(#loc)
} loc(#loc)

// CHECK: #[[LOC_SP:.+]] = loc(fused<#{{.*}}>[#[[LOC_RAW]]])
// CHECK: #[[CALL_LOC:.+]] = #debuginfo.call_loc<#[[LOC_SP]]>
// CHECK: #[[LOC_STAGECLOSURE]] = loc(fused<#[[CALL_LOC]]>[#[[LOC_RAW]]])
