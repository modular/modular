# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.dir
# RUN: mojo package %S/../package/test_package -o %t.dir/test_binary_package.mojoc
# RUN: not mojo run -I %t.dir %s 2>&1 | FileCheck %s

# COM: This will import a package with the wrong case, so we expect an error.
# CHECK: error: unable to locate module 'TEST_BINARY_PACKAGE'
from TEST_BINARY_PACKAGE.inner1.myfile import print10


def main() raises:
    # CHECK: 10
    print10()
