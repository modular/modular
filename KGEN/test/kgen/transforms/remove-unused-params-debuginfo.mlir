// RUN: kgen-opt --split-input-file --remove-unused-params --mlir-print-debuginfo %s  | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"basic_arg_remove_1">>  : !debuginfo.subroutine<(!pop.scalar<dt>, !kgen.pointer<T>, !kgen.pointer<scalar<dt>>) -> (): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":1:1])

// CHECK: kgen.generator @basic_arg_remove_1_REMOVED_ARG(%[[ARG0:.+]]: !kgen.pointer<none>
kgen.generator @basic_arg_remove_1<dt: dtype, T: type>(%arg0: !pop.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  pop.load %arg1 : !kgen.pointer<none> loc(#loc)
  pop.load %arg2 : !kgen.pointer<none> loc(#loc)
  kgen.return loc(#loc)
// CHECK: } loc(#[[LOC:.+]])
} loc(#loc)

kgen.generator @user<dt: dtype, T: type>(%arg0: !pop.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  kgen.call @basic_arg_remove_1<:dtype dt, :type T>(%arg0, %arg1, %arg2) : (!pop.scalar<dt>, !kgen.pointer<none>, !kgen.pointer<none>) -> ()
  kgen.return
}

// CHECK: ![[UNSPECIFIED:.+]] = !debuginfo.unspecified<"optimized out">
// CHECK: ![[SUBROUTINE:.+]] = !debuginfo.subroutine<(![[UNSPECIFIED]], !kgen.pointer<*?>, !kgen.pointer<scalar<*?>>) -> (): DW_CC_normal>
// CHECK: #[[SP:.+]] = #debuginfo.subprogram<name = <"basic_arg_remove_1">, linkageName = "basic_arg_remove_1_REMOVED_ARG"> : ![[SUBROUTINE]]
// CHECK: #[[LOC]] = loc(fused<#[[SP]]>
