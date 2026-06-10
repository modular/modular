# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mkdir -p %t.closure-dir
# RUN: mojo precompile %S/inputs/closure -o %t.closure-dir/closure.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.closure-dir %s | FileCheck %s


# CHECK-LABEL: lit.struct.decl @Box
# CHECK: lit.fn @"__init__{{.*}}"{{.*}}*, %move:

from closure import Box


def main() raises:
    var box = Box()

    def use_box() raises {var box^}:
        pass
