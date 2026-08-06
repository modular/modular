# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A directory whose name differs only by case is not a namespace portion:
# portion matching stays case-exact even on case-insensitive filesystems.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo %t.dir/two/FOO
# RUN: echo "def hello():" > %t.dir/one/foo/test.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/test.mojo
# RUN: echo "def hello2():" > %t.dir/two/FOO/test2.mojo
# RUN: echo "    pass" >> %t.dir/two/FOO/test2.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/one -I=%t.dir/two %s

import foo.test

# expected-error @+1 {{unable to locate module 'test2'}}
import foo.test2
