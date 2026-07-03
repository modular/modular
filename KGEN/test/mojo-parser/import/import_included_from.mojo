# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Regression test: a diagnostic about a declaration reached through imports
# prints an "Included from" stack rooted at the user's import and walking
# through each re-exporting module/package down to the definition.
#
# `parametric_fn` is defined in test_nested_package/module.mojo; using it
# without binding its parameter is a parameter-inference error whose note points
# at that declaration, so it exercises the include stack independently of
# overload resolution.

# RUN: not %parse-mojo-isolated -split-input-file -I=%S/inputs %s 2>&1 | FileCheck %s

# Reached through an intermediate module: test_package/module.mojo re-exports
# `parametric_fn` from test_nested_package/module.mojo. The stack flows
#   this file -> test_package/module.mojo (re-export) -> the defining module.
from test_package.module import parametric_fn

def main():
    _ = parametric_fn()

# CHECK: error: invalid call to 'parametric_fn': failed to infer parameter 'n'
# CHECK: Included from {{.*}}import_included_from.mojo
# CHECK-NEXT: Included from {{.*}}test_package{{.*}}module.mojo:{{[0-9]+}}:
# CHECK-NEXT: test_nested_package{{.*}}module.mojo:{{[0-9]+}}:{{[0-9]+}}: note: function declared here

# // -----

# Reached through a package __init__ re-export: test_nested_package/__init__.mojo
# re-exports `parametric_fn`. The stack is rooted at this file's import (not at
# whatever lazy lookup first opened __init__).
from test_package.test_nested_package import parametric_fn

def main():
    _ = parametric_fn()

# CHECK: error: invalid call to 'parametric_fn': failed to infer parameter 'n'
# CHECK: Included from {{.*}}import_included_from.mojo
# CHECK-NEXT: Included from {{.*}}test_nested_package{{.*}}__init__.mojo:{{[0-9]+}}:
# CHECK-NEXT: test_nested_package{{.*}}module.mojo:{{[0-9]+}}:{{[0-9]+}}: note: function declared here

# // -----

# Reached through a package __init__ that re-exports a sub-package with `*`:
# star_reexport_package/__init__.mojo does `from .subpkg import *`, so the stack
# flows through *both* package __init__s down to the definition.
#   this file -> star_reexport_package/__init__.mojo (import *)
#             -> star_reexport_package/subpkg/__init__.mojo (re-export) -> leaf
from star_reexport_package import needs_param

def main():
    _ = needs_param()

# CHECK: error: invalid call to 'needs_param': failed to infer parameter 'n'
# CHECK: Included from {{.*}}import_included_from.mojo
# CHECK-NEXT: Included from {{.*}}star_reexport_package{{.*}}__init__.mojo:{{[0-9]+}}:
# CHECK-NEXT: Included from {{.*}}star_reexport_package{{.*}}subpkg{{.*}}__init__.mojo:{{[0-9]+}}:
# CHECK-NEXT: star_reexport_package{{.*}}subpkg{{.*}}leaf.mojo:{{[0-9]+}}:{{[0-9]+}}: note: function declared here
