# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo build-project --diagnostic-format json --doesnt-exist 2>&1 | FileCheck %s
# CHECK: {"kind":"error","message":"unrecognized argument '--doesnt-exist'"}
