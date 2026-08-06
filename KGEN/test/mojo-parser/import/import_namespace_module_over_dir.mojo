# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Within a namespace, a module in any portion beats a plain directory in an
# earlier portion: plain directories carry no marker, so a stray non-Mojo
# directory must not shadow a real module.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo/util %t.dir/two/foo
# RUN: echo "def util_fn():" > %t.dir/two/foo/util.mojo
# RUN: echo "    pass" >> %t.dir/two/foo/util.mojo
# RUN: %parse-mojo-isolated -I=%t.dir/one -I=%t.dir/two %s

from foo.util import util_fn
