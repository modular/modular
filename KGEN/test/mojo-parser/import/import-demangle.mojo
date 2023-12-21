# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -mojo-disable-builtins -import-mojo -I %S %s | FileCheck %s
# RUN: kgen-translate -mojo-disable-builtins -import-mojo -I %S %s | FileCheck %s

# COM: Run it twice to ensure it works on a cache hit.

from test_package.module import `use()weird[]`

# CHECK: lit.package @"$test_package"
# CHECK-NEXT: lit.file_module @"$module"
# CHECK: lit.struct.decl @"weird()struct[]"
# CHECK: lit.func @"use()weird[]()"


fn weird_struct():
    _ = `use()weird[]`()
