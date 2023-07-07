# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# We can run this file with various targets.
# RUN: not mojo-driver run -target-triple not-a-valid-target %s 2>&1 | FileCheck %s
# CHECK: no target exists for 'not-a-valid-target'
