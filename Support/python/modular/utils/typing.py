##=== typing.py -----------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
# TODO: code in modular should use this instead of the builtin module


from sys import version_info
from typing import *

if version_info.minor <= 8:
    from typing import Iterable, Iterator
else:
    from collections.abc import Iterable, Iterator

del version_info
