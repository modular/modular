# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
A package whose directory name contains a period, importable only via an
escaped identifier (`dotted.pkg`). Used to test dotted-path resolution.
"""


def pkg_init_fn() -> Int:
    return 1
