# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
