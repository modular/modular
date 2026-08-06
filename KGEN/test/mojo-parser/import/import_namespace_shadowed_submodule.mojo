# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A module shadows a same-named plain directory across portions completely:
# the directory's contents are unreachable through the name, in either
# root order (Python's rule; with util.py present, 'foo.util' is
# not a package).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo/util %t.dir/two/foo
# RUN: echo "def inner_fn():" > %t.dir/one/foo/util/inner.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/util/inner.mojo
# RUN: echo "def util_fn():" > %t.dir/two/foo/util.mojo
# RUN: echo "    pass" >> %t.dir/two/foo/util.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/one -I=%t.dir/two %s
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/two -I=%t.dir/one %s

# expected-error @+1 {{'util' is a module, not a package; it has no nested module or package 'inner'}}
import foo.util.inner
