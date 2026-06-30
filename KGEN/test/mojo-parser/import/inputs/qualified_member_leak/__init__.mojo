# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""A package that re-exports a symbol and contains sibling modules.

`shared_fn` is re-exported here so it lives in __init__'s scope, and `helpers`
is a sibling module. Neither is a member of the `other` module - a *qualified*
access like `other.shared_fn` / `other.helpers` must therefore be a hard error,
not a deprecated-intra-package resolution against this package.
"""

from .helpers import shared_fn
