# ===- typing.py ----------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import version_info
from typing import *  # noqa: F403

if version_info.minor <= 8:
    from typing import IO, BinaryIO, Match, Pattern, TextIO
else:
    from collections.abc import Iterable, Iterator

    Tuple = tuple

del version_info
