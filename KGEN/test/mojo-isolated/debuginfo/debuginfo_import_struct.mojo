# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I %S -debug-level full -mlir-print-debuginfo %s | FileCheck %s

from debuginfo_module import VeryUniqueStruct

# CHECK-DAG: #[[FILE:file[0-9]+]] = #debuginfo.file<"[[FILENAME:.*debuginfo_module.mojo]]" in
# CHECK-DAG: !VeryUniqueStruct

# CHECK-DAG: lit.struct.decl @VeryUniqueStruct
# CHECK-DAG: lit.struct.field very_unique_field : index loc(#[[LOC:loc[0-9]+]])
# CHECK-DAG: lit.func @"very_unique_func{{.*}}"(%C-3PO: index loc(#[[LINE_LOC:.*]])

# CHECK-DAG: #[[LOC]] = loc(fused<#[[FILE]]>[#loc{{[0-9]+}}])


fn caller():
    var y = VeryUniqueStruct.very_unique_func(__mlir_attr.`0 : index`)
