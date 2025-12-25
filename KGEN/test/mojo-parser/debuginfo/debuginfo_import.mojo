# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I %S -debug-level full -mlir-print-debuginfo %s | FileCheck %s

from debuginfo_module import imported_fn

# Check that we properly generate functions that get resolved within other functions.
# This is mostly checking that the scope of the nested function is not another function.

# CHECK-DAG: #[[CALLED_STRUCT_BOUND:.*]] = #debuginfo.source_name<(struct)"CalledStruct{{.*}}param{{.*}}"
# CHECK-DAG: #[[CALLED_STRUCT:.*]] = #debuginfo.source_name<(struct)"CalledStruct"[<"index">] from <(module)"debuginfo_import">>
# CHECK-DAG: #test_name = #debuginfo.source_name<(fn)"test"(#[[CALLED_STRUCT_BOUND]]) from #[[CALLED_STRUCT]]>
# CHECK-DAG: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = {{.*}}, sourceName = #test_name, linkageName = "test({{.*}}::CalledStruct[{{.*}}])"


struct CalledStruct[param: __mlir_type.index]:
    fn test(self):
        imported_fn()


fn callerFn[rows: __mlir_type.index](arg0: CalledStruct[rows]):
    return arg0.test()
