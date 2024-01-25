# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.dir
# RUN: mojo package %mojo_cpu_build_arch %S/test_package -o %t.dir/test_binary_package.mojopkg
# RUN: mojo run %mojo_cpu_build_arch -I %t.dir %s | FileCheck %s

from test_binary_package.inner1.myfile import print10
from test_binary_package.inner1 import myfile_copy


def main():
    # CHECK: 10
    # CHECK: 10
    print10()
    myfile_copy.print10()
