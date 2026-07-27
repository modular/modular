# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""A module (`plain_mod.dot`) whose name extends `plain_mod` with a dotted
suffix; the two must never alias."""


def from_dotted() -> Int:
    return 6
