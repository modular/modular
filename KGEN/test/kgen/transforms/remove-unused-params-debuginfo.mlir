// RUN: kgen-opt --split-input-file --remove-unused-params --mlir-print-debuginfo %s  | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"basic_arg_remove_1">>  : !debuginfo.subroutine<(!pop.scalar<dt>, !kgen.pointer<T>, !kgen.pointer<scalar<dt>>) -> (): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":1:1])

#otherSP = #debuginfo.subprogram<name = <"param_inlined_fn">>  : !debuginfo.subroutine<(!kgen.pointer<T>) -> (): DW_CC_normal>
#otherLoc = loc(fused<#otherSP>["foo.mlir":2:2])
#inlinedLoc = loc(callsite(#otherLoc at #loc))

// CHECK: kgen.generator @basic_arg_remove_1_REMOVED_ARG(%[[ARG0:.+]]: !kgen.pointer<none>
kgen.generator @basic_arg_remove_1<dt: dtype, T: type>(%arg0: !pop.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  pop.load %arg1 : !kgen.pointer<none> loc(#loc)
  pop.load %arg2 : !kgen.pointer<none> loc(#loc)
  // CHECK: kgen.return loc(#[[INLINED_LOC:.+]])
  kgen.return loc(#inlinedLoc)
// CHECK: } loc(#[[LOC:.+]])
} loc(#loc)

kgen.generator @user<dt: dtype, T: type>(%arg0: !pop.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  kgen.call @basic_arg_remove_1<:dtype dt, :type T>(%arg0, %arg1, %arg2) : (!pop.scalar<dt>, !kgen.pointer<none>, !kgen.pointer<none>) -> ()
  kgen.return
}

// CHECK-DAG: ![[UNSPECIFIED:.+]] = !debuginfo.unspecified<"optimized out">
// CHECK-DAG: ![[SUBROUTINE:.+]] = !debuginfo.subroutine<(![[UNSPECIFIED]], !kgen.pointer<*?>, !kgen.pointer<scalar<*?>>) -> (): DW_CC_normal>
// CHECK-DAG: #[[SP:.+]] = #debuginfo.subprogram<name = <"basic_arg_remove_1">, linkageName = "basic_arg_remove_1_REMOVED_ARG"> : ![[SUBROUTINE]]
// CHECK-DAG: #[[LOC]] = loc(fused<#[[SP]]>


// CHECK-DAG: ![[OTHER_SUBROUTINE:.+]] = !debuginfo.subroutine<(!kgen.pointer<*?>) -> (): DW_CC_normal>
// CHECK-DAG: #[[OTHER_SP:.+]] = #debuginfo.subprogram<name = <"param_inlined_fn">> : ![[OTHER_SUBROUTINE]]
// CHECK-DAG: #[[OTHER_LOC:.+]] = loc(fused<#[[OTHER_SP]]>
// CHECK-DAG: #[[INLINED_LOC]] = loc(callsite(#[[OTHER_LOC]] at
