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
#   Path:   tests/data/simple_cases/fmtskip5.py
#
# ===----------------------------------------------------------------------=== #

a, b, c = 3, 4, 5
if (
    a == 3
    and b    != 9  # fmt: skip
    and c is not None
):
    print("I'm good!")
else:
    print("I'm bad")


# output

a, b, c = 3, 4, 5
if (
    a == 3
    and b    != 9  # fmt: skip
    and c is not None
):
    print("I'm good!")
else:
    print("I'm bad")
