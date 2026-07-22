# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Named by a multi-component relative wildcard in the package __init__."""


# Re-exported by `from .deep.inner import *` in the package __init__, but the
# walk resolves wildcards by leaf name only, so this is NOT found (limitation).
def multicomp_wildcard_symbol():
    pass
