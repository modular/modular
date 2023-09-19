# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Run parsing twice to ensure the cache is populated.
# RUN: kgen-translate -import-mojo -I=%S %s
# RUN: kgen-translate -import-mojo -I=%S %s | FileCheck %s

from imported_cached_module import StringLiteralAlias, global_variable

# CHECK-LABEL: lit.func @"assign_from()"
fn assign_from():
    # CHECK: !StringLiteral = <{{.*}}"foobar"
    let foo = StringLiteralAlias
    # CHECK: lit.globalvar.ref {{.*}}@global_variable
    let bar = global_variable
