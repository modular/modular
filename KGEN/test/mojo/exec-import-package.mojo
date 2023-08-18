# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.dir
# RUN: mojo package %mojo_cpu_build_arch %S/test_package --name test_binary_package -o %t.dir/test_binary_package.mojopkg
# RUN: mojo run %mojo_cpu_build_arch -I %t.dir %s | FileCheck %s

from test_binary_package.inner1.myfile import print10


def main() -> None:
    # CHECK: 10
    print10()
