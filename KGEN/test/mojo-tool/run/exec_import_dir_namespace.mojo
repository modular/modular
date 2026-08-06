# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# One plain directory name can span several import roots, like a Python
# namespace package: submodules resolve under every root providing the
# directory, not just the first, and nested plain directories merge the same
# way.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo/bar %t.dir/two/foo/bar
# RUN: cp %S/inputs/namespace/test.mojo %t.dir/one/foo/test.mojo
# RUN: cp %S/inputs/namespace/test2.mojo %t.dir/two/foo/test2.mojo
# RUN: cp %S/inputs/namespace/baz.mojo %t.dir/one/foo/bar/baz.mojo
# RUN: cp %S/inputs/namespace/qux.mojo %t.dir/two/foo/bar/qux.mojo
# RUN: mojo run -I %t.dir/one -I %t.dir/two %s | FileCheck %s

# CHECK: namespace test one
# CHECK-NEXT: namespace test2 two
# CHECK-NEXT: namespace baz one
# CHECK-NEXT: namespace qux two

import foo.test
import foo.test2
import foo.bar.baz
import foo.bar.qux


def main():
    foo.test.hello()
    foo.test2.hello2()
    foo.bar.baz.baz_fn()
    foo.bar.qux.qux_fn()
