# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tests that the `-o` output file name of the `.mojopkg` determines the name by
# which the package can be imported -- the original directory name doesn't
# matter.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %T/package-rename && mkdir -p %T/package-rename
# RUN: mojo package %S/inputs/test_package -o %T/package-rename/renamed-package.mojopkg
# RUN: %mojo -I %T/package-rename %s | FileCheck %s

from `renamed-package`.module import identity


fn main():
    # CHECK: hi
    print(identity("hi"))
