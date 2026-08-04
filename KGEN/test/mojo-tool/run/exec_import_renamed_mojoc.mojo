# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A renamed precompiled package is self-contained: it loads with the source
# tree gone, including when its modules use absolute self-imports (which
# once recorded a dangling self-dependency that forced a source re-parse).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/src/pkg %t.dir/lib
# RUN: echo "# pkg" > %t.dir/src/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/src/pkg/a.mojo
# RUN: echo "    print(42)" >> %t.dir/src/pkg/a.mojo
# RUN: echo "from pkg.a import afn" > %t.dir/src/pkg/b.mojo
# RUN: echo "def bfn():" >> %t.dir/src/pkg/b.mojo
# RUN: echo "    afn()" >> %t.dir/src/pkg/b.mojo
# RUN: mojo precompile %t.dir/src/pkg -o %t.dir/lib/renamed_pkg.mojoc
# RUN: rm -rf %t.dir/src
# RUN: mojo run -I %t.dir/lib %s | FileCheck %s

# CHECK: 42

from renamed_pkg.b import bfn


def main():
    bfn()
