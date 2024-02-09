# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s

# COM: These tests are in a separate file only because mblack fails to parse them.
# TODO(#31345): Move these back to global-var-let.mojo when mblack is fixed

# # CHECK: lit.globalvar.decl export @exported_alias {{.*}} {linkageName = "exported_global"}
@export("exported_global")
var exported_alias = `4`

# # CHECK: lit.globalvar.decl export C @exported_global_var {{.*}} {linkageName = "exported_global_var"}
@export(ABI="C")
var exported_global_var = `5`
