# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo package %S/test_package -o test.mojonot 2>&1 | FileCheck %s
# RUN: not mojo package %S/test_package -o not-a-directory/ 2>&1 | FileCheck %s
# CHECK: output path must have a '.mojopkg' or '.📦' extension
