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
#   Path:   tests/data/simple_cases/trailing_comma_optional_parens2.py
#
# ===----------------------------------------------------------------------=== #

if e123456.get_tk_patchlevel() >= (8, 6, 0, "final") or (
    8,
    5,
    8,
) <= get_tk_patchlevel() < (8, 6):
    pass

# output

if e123456.get_tk_patchlevel() >= (8, 6, 0, "final") or (
    8,
    5,
    8,
) <= get_tk_patchlevel() < (8, 6):
    pass
