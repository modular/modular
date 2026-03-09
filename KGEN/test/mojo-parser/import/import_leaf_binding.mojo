# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S %s

# Test that `import a.b` emits a deprecation warning when the leaf name `b` is
# used unqualified. Also verify that `import a.b as b` and `from a import b` do
# NOT emit the warning.

# Leaf import without 'as': should warn

import test_package.module

fn test_leaf_import():
    # expected-warning @below {{use of unqualified 'module' is deprecated; use fully-qualified 'test_package.module' here, or a different import statement, e.g., 'import test_package.module as module' or 'from test_package import module'}}
    module.function()
    # should not warn
    test_package.module.function()

# // -----

# Import with 'as': should NOT warn

import test_package.module as mod

fn test_as_import():
    mod.function()

# 'from' import: should NOT warn

from test_package import module as mod2

fn test_from_import():
    mod2.function()
