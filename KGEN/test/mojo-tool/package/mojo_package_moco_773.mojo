# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo package %S/test_package_moco_773 -o test_package_moco_773.mojoc 2>&1 | FileCheck %s
# CHECK: foo.mojo:12:31: error: functions must not declare both an 'out' argument and a return type
