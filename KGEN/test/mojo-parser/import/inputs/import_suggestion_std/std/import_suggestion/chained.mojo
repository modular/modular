# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Reached only through a wildcard nested inside another wildcard re-export."""


# Exposed to the package only via wild_chain's own `from .chained import *`, so
# the one-level walk does NOT find it (limitation).
def transitive_symbol():
    pass
