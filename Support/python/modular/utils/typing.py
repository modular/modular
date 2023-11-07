# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Typing utility library

This is a drop-in replacement for the Python standard library's typing module.
Currently, the sole purpose of this library is to hide API changes that occurred
between different Python major versions, so that scripts don't need to deal
with version checking. Modular developers should use this module over the
built-in typing module.
"""

from enum import Enum
from sys import version_info
from typing import *  # noqa: F403

if version_info.minor <= 8:
    from typing import IO, BinaryIO, Match, Pattern, TextIO
else:
    from collections.abc import Iterable, Iterator

    Tuple = tuple

del version_info
