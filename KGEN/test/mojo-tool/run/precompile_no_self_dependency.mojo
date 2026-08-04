# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A package never records itself as a link dependency, even when a module
# absolutely imports its own package (which materializes a duplicate
# package op during precompilation).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "from pkg.a import afn" > %t.dir/pkg/b.mojo
# RUN: mojo precompile %t.dir/pkg -o %t.dir/renamed_pkg.mojoc
# RUN: kgen-opt %t.dir/renamed_pkg.mojoc | FileCheck %s --implicit-check-not link.dependencies

# CHECK: lit.package @renamed_pkg
