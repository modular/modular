# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Two roots providing the same module under one namespace directory is an
# error (stricter than Python, which silently takes path order).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo %t.dir/two/foo
# RUN: echo "def hello():" > %t.dir/one/foo/test.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/test.mojo
# RUN: cp %t.dir/one/foo/test.mojo %t.dir/two/foo/test.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/one -I=%t.dir/two %s

# expected-error @+1 {{ambiguous import 'test': found}}
import foo.test
