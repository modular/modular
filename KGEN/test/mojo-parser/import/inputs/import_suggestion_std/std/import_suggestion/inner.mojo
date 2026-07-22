# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Direct child module whose leaf name (`inner`) collides with the final
component of the package __init__'s `from .deep.inner import *`. Not re-exported
by any __init__, so its symbol is not public API and must get no suggestion. It
exists to pin the leaf-collision bug: resolving that multi-component wildcard by
leaf name alone would false-match this unrelated direct child."""


def collision_symbol():
    pass
