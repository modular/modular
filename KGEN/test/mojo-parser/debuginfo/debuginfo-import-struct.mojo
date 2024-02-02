# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I %S -debug-level full -mlir-print-debuginfo %s | FileCheck %s

from debuginfo_module import VeryUniqueStruct

# CHECK-DAG: #[[FILE:file[0-9]+]] = #debuginfo.file<"[[FILENAME:.*debuginfo_module.mojo]]" in
# CHECK-DAG: #[[LINE_LOC:loc[0-9]+]] = loc("{{.*}}debuginfo_module.mojo":
# CHECK-DAG: #[[LINE_LOC2:loc[0-9]+]] = loc("{{.*}}debuginfo_module.mojo":
# CHECK: !VeryUniqueStruct

# CHECK-DAG: #[[LOCAL_VAR:local_variable[0-9]*]] = #debuginfo.local_variable<scope = #[[SP:subprogram[0-9]+]], name = "C-3PO", file = #[[FILE]],

# CHECK-DAG: lit.struct.decl @VeryUniqueStruct
# CHECK-DAG: lit.struct.field very_unique_field : index loc(#[[LOC:loc[0-9]+]])
# CHECK-DAG: lit.func @"very_unique_func{{.*}}"(%C-3PO: index loc(#[[LINE_LOC2]]
# CHECK-DAG: debuginfo.value #[[LOCAL_VAR]] = %C-3PO : index loc(#[[VALUE_LOC:loc[0-9]+]])

# CHECK-DAG: #[[LOC]] = loc(fused<#[[FILE]]>[#loc{{[0-9]+}}])
# CHECK-DAG: #[[VALUE_LOC]] = loc(fused<#[[SP]]>[#[[LINE_LOC2]]])


fn caller():
    let y = VeryUniqueStruct.very_unique_func(__mlir_attr.`0 : index`)
