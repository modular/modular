# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Import `foo` along two paths that both resolve to `source.foo`: directly and
# through `relay`. Under keep-and-gate each path becomes its own resolved
# `lit.import` gate, so this module's scope ends up holding two gates for `foo`.
# Using `foo` forces both gates to resolve during precompile, so they are
# serialized as *resolved* gates. Reloading this bytecode must collapse the two
# gates rather than reject the second as an "invalid redefinition".

from .relay import foo
from .source import foo


def use_foo() -> Int:
    return foo()
