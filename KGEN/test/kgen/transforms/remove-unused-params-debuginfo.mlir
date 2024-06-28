// RUN: kgen-opt --split-input-file --remove-unused-params --mlir-print-debuginfo %s  | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"basic_arg_remove_1">>  : !debuginfo.subroutine<(index, !kgen.pointer<T>) -> (): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":1:1])

// CHECK-LABEL: kgen.generator @basic_arg_remove_1
kgen.generator @basic_arg_remove_1<T: type>(%arg0: index, %arg1: !kgen.pointer<none>) {
  pop.load %arg1 : !kgen.pointer<none> loc(#loc)
  kgen.return loc(#loc)
// CHECK: } loc(#[[LOC:.+]])
} loc(#loc)

// CHECK: ![[SUBROUTINE:.+]] = !debuginfo.subroutine<(index, !kgen.pointer<*?>) -> (): DW_CC_normal>
// CHECK: #[[SP:.+]] = #debuginfo.subprogram<name = <"basic_arg_remove_1">> : ![[SUBROUTINE]]
// CHECK: #[[LOC]] = loc(fused<#[[SP]]>
