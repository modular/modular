# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tests that the `-o` output file name of the `.mojoc` determines the name by
# which the package can be imported -- the original directory name doesn't
# matter.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.package-rename && mkdir -p %t.package-rename
# RUN: mojo precompile %S/inputs/test_package -o %t.package-rename/renamed-package.mojoc
# RUN: %mojo -I %t.package-rename %s | FileCheck %s

from `renamed-package`.module import identity


def main():
    # CHECK: hi
    print(identity("hi"))
