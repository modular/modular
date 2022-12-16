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
#   Path:   tests/data/miscellaneous/pattern_matching_invalid.py
#
# ===----------------------------------------------------------------------=== #

# First match, no errors
match something:
    case bla():
        pass

# Problem on line 10
match invalid_case:
    case valid_case:
        pass
    case a := b:
        pass
    case valid_case:
        pass

# No problems either
match something:
    case bla():
        pass
