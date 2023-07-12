# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -I %S -split-input-file -debug-level=full -mlir-print-debuginfo %s | FileCheck %s

from imported_module import imported_fn

# Check that we properly generate functions that get resolved within other functions.
# This is mostly checking that the scope of the nested function is not another function.

# CHECK-DAG: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "test", linkageName = "test($parser-debuginfo::CalledStruct[{{.*}}param])"


struct CalledStruct[param: __mlir_type.index]:
    fn test(self):
        imported_fn()


fn callerFn[rows: __mlir_type.index](arg0: CalledStruct[rows]):
    return arg0.test()


# Check single file debug info generation.

# CHECK-DAG: ![[INT_TYPE:.*]] = !debuginfo.unresolved<!kgen.declref<{{.*}}@"$Int"::@Int>>
# CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(!kgen.declref<{{.*}}@"$Int"::@Int>, !kgen.declref<{{.*}}@"$Int"::@Int>) -> (!kgen.declref<{{.*}}@"$Int"::@Int>): DW_CC_normal>
# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "power", linkageName = "power({{.*}}$Int::Int,{{.*}}$Int::Int)", file = #{{.*}}, line = 35, scopeLine = 35, subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK-DAG: #[[LHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "lhs", file = #{{.*}}, line = 35, arg = 1> : ![[INT_TYPE]]
# CHECK-DAG: #[[RHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "rhs", file = #{{.*}}, line = 35, arg = 2> : ![[INT_TYPE]]

# CHECK-LABEL: lit.func @"power({{.*}}$Int::Int,{{.*}}$Int::Int)"(
fn power(lhs: Int, rhs: Int) -> Int:
    # CHECK: debuginfo.value #[[LHS_VAR]] = %lhs
    # CHECK: debuginfo.value #[[RHS_VAR]] = %rhs
    return lhs


# // -----

# CHECK-DAG: #[[LOCAL_VAR_I:.*]] = #debuginfo.local_variable<scope = #[[FOR_SP:.*]], name = "i", {{.*}}, line = 8, arg = 1

# CHECK-LABEL: lit.func @"structured_for_loop()"
fn structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 = %idx0 : index) -> index {
    __mlir_region loop_body(i: __mlir_type.index):
        # CHECK-NEXT: debuginfo.value #[[LOCAL_VAR_I]] = %arg0 : index
        # CHECK-NEXT: %idx1 = index.constant 1
        # CHECK-NEXT: %1 = index.add %arg0, %idx1 loc(#[[FOR_ADD_LOC:.*]])
        # CHECK-NEXT: hlcf.continue %1 : index loc(#[[FOR_YIELD_LOC:.*]])
        __mlir_op.`hlcf.continue`(__mlir_op.`index.add`(i, (1).value))

    # CHECK: lit.return %0 : index
    return __mlir_op.`hlcf.loop`[
        _type : __mlir_type.index, _region : "loop_body".value
    ]((0).value)


# // -----

from imported_module import VeryUniqueStruct

# CHECK-DAG: #[[FILE:file[0-9]+]] = #debuginfo.file<"[[FILENAME:.*imported_module.mojo]]" in

# CHECK-DAG: #[[LOCAL_VAR:local_variable[0-9]+]] = #debuginfo.local_variable<scope = #[[SP:subprogram[0-9]+]], name = "very_unique_arg", file = #[[FILE]],

# CHECK-DAG: lit.struct.decl @VeryUniqueStruct
# CHECK-DAG: lit.struct.field very_unique_field : index loc(#[[LOC:loc[0-9]+]])
# CHECK-DAG: lit.func @"very_unique_func($Builtin::$Int::Int)"(%very_unique_arg: !kgen.declref<@"$Builtin"::@"$Int"::@Int> loc("[[FILENAME]]"
# CHECK-DAG: debuginfo.value #[[LOCAL_VAR]] = %very_unique_arg : !kgen.declref<@"$Builtin"::@"$Int"::@Int> loc(#[[VALUE_LOC:loc[0-9]+]])

# CHECK-DAG: #[[LOC]] = loc(fused<#[[FILE]]>[#loc{{[0-9]+}}])
# CHECK-DAG: #[[VALUE_LOC]] = loc(fused<#[[SP]]>[#[[LINE_LOC:loc[0-9]+]]])
# CHECK-DAG: #[[LINE_LOC]] = loc("[[FILENAME]]"

fn caller():
    let y = VeryUniqueStruct.very_unique_func(0)
