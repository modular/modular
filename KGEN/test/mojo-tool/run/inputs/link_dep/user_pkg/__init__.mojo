# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""A package whose precompiled form records `dotted.dep` as a link
dependency."""

from `dotted.dep` import dep_value


def user_value() -> Int:
    return dep_value()
