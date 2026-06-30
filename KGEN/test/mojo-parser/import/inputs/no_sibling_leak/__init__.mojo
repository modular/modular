# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""A package whose modules must import their siblings explicitly.

The re-export below makes `producer` reachable as `no_sibling_leak.producer`
from outside, but a *sibling* module still cannot see it without its own
import - re-exports live in __init__'s scope, not the package scope a
contained file walks up into.
"""

from . import producer
from .producer import reexported_fn
