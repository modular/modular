# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Modules and packages are importable through plain source directories (no
# __init__.mojo), so their API surface must show up in generated docs too.
# `dir` here is a plain directory inside the package.

# RUN: kgen-doc %S/test_dir_package | FileCheck %s

# CHECK:  "kind": "package"
# CHECK:  "modules":
# CHECK:    "name": "__init__"
# CHECK:  "name": "test_dir_package"
# CHECK:  "packages":
# CHECK:    "name": "dir_fn"
# CHECK:    "summary": "Does directory things."
# CHECK:    "name": "module"
# CHECK:    "summary": "A module inside a plain directory."
# CHECK:    "name": "dir"
# CHECK:  "summary": "This is a package with a plain source directory."
