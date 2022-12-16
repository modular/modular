# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/miscellaneous/docstring_preview_no_string_normalization.py
#
# ===----------------------------------------------------------------------=== #


def do_not_touch_this_prefix():
    R"""There was a bug where docstring prefixes would be normalized even with -S."""


def do_not_touch_this_prefix2():
    Rf"There was a bug where docstring prefixes would be normalized even with -S."


def do_not_touch_this_prefix3():
    """There was a bug where docstring prefixes would be normalized even with -S."""
