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
#   Path:   tests/data/simple_cases/docstring_no_extra_empty_line_before_eof.py
#
# ===----------------------------------------------------------------------=== #

# Make sure when the file ends with class's docstring,
# It doesn't add extra blank lines.
class ClassWithDocstring:
    """A docstring."""
