# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo package %S/test_package -o test.mojonot 2>&1 | FileCheck %s
# RUN: not mojo package %S/test_package -o not-a-directory/ 2>&1 | FileCheck %s
# RUN: not mojo package --diagnostic-format json %S/test_package \
# RUN:   -o test.mojonot 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG
# CHECK: output path must have a '.mojopkg' or '.📦' extension
# CHECK-DIAG: "kind":"error","message":"output path must have a '.mojopkg' or '.📦' extension"}
