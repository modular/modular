# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-doc %S/test_package | FileCheck %s

# CHECK:  "description": "This is a description.",
# CHECK:  "kind": "package"
# CHECK:  "modules":
# CHECK:    "name": "__init__"
# CHECK:  "name": "test_package"
# CHECK:  "packages":
# CHECK:    "name": "inner1"
# CHECK:    "name": "inner2"
# CHECK:  "summary": "This is a test package."
