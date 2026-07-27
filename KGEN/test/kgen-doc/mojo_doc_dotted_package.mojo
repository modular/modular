# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-doc %S/dotted.pkg | FileCheck %s

# A package directory (and sub-package/module) whose name contains periods
# keeps its whole name in the generated docs: the periods are part of the
# name, not an extension to strip.

# CHECK:  "kind": "package"
# CHECK:  "modules":
# CHECK:    "name": "__init__"
# CHECK:    "name": "module.with.dots"
# CHECK:  "name": "dotted.pkg"
# CHECK:  "packages":
# CHECK:    "name": "sub.pkg"
# CHECK:  "summary": "This is a dotted package."
