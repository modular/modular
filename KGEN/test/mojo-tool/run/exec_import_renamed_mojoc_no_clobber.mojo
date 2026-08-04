# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A renamed precompiled package claims only its file's name: its internal
# absolute self-references bind to the package itself, and a separate module
# with the package's original source name resolves independently.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/src/foo %t.dir/lib
# RUN: echo "# pkg" > %t.dir/src/foo/__init__.mojo
# RUN: echo "def afn():" > %t.dir/src/foo/a.mojo
# RUN: echo "    print(1000)" >> %t.dir/src/foo/a.mojo
# RUN: echo "from foo.a import afn" > %t.dir/src/foo/b.mojo
# RUN: echo "def bfn():" >> %t.dir/src/foo/b.mojo
# RUN: echo "    afn()" >> %t.dir/src/foo/b.mojo
# RUN: mojo precompile %t.dir/src/foo -o %t.dir/lib/bar.mojoc
# RUN: rm -rf %t.dir/src
# RUN: echo "def foo_fn():" > %t.dir/lib/foo.mojo
# RUN: echo "    print(2000)" >> %t.dir/lib/foo.mojo
# RUN: mojo run -I %t.dir/lib %s | FileCheck %s

# CHECK: 1000
# CHECK-NEXT: 2000

import foo
from bar.b import bfn


def main():
    bfn()
    foo.foo_fn()
