# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Aliased from-import of a sibling submodule (mirrors `from std.sys import
# _libc as libc`): this binds a gated ImportOp under `impl` over the `_impl`
# submodule. The gate must not be serialized alongside its now-dead
# `unresolved_import` placeholder, or reloading this package fails with
# "invalid redefinition of 'impl'".
from . import _impl as impl


def api_value() -> Int:
    return impl.impl_value()
