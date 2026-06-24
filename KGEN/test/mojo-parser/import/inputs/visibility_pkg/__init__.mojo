# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""A package re-exporting `shown`, and `sub` only under the alias `visible_sub`.

`sub` stays gated under its own name; it is reachable only through the alias.
"""

from . import shown
from . import sub as visible_sub
