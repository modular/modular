# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Regression test: the MOCO-2845 self-named-submodule fix must not leak into
# top-level `-I` resolution, where a same-named `.mojoc`/`.mojo` pair is an
# ordinary collision and `.mojoc` must still win.

# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: cp -r %S/inputs/top_level_name_collision/foo %t.dir/foo
# RUN: mojo precompile %S/inputs/top_level_name_collision/foo_pkg_src -o %t.dir/foo/foo.mojoc
# RUN: mojo run -I %t.dir/foo %s | FileCheck %s

from foo import describe


def main():
    # CHECK: precompiled
    describe()
