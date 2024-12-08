# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo package %S/test_package_moco_773 -o test_package_moco_773.mojopkg 2>&1 | FileCheck %s
# CHECK: foo.mojo:12:21: error: function cannot have both an 'out' argument and an explicit result type
