# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This test checks for diagnostic quality when source location information
# isn't available. It does so by precompiling a package and deleting the
# original source. In such cases the compiler should be able to pretty-print
# useful information for the user from the in-memory MLIR constructs.

# Copy the package to a temporary directory
# RUN: mkdir -p %t
# RUN: cp -r %S/inputs/diags_package %t

# Precompile the package
# RUN: mojo precompile %t/diags_package -o %t/diags_package.mojoc

# Remove the source
# RUN: rm -r %t/diags_package

# RUN: not %mojo-build -I %t %s 2>&1 | FileCheck %s

import diags_package


def main():
    # CHECK: error: invalid call to 'fn_missing_constraint': violated constraint
    # FIXME: Improve the pretty-printing of this constraint
    # CHECK: note: constraint declared here as '($0 > 0)' evaluated to False, expected '(n > 0)'
    # CHECK: note: function declared here:
    # CHECK-NEXT: def fn_missing_constraint[n: Int]() where (n > 0)
    diags_package.fn_missing_constraint[0]()
