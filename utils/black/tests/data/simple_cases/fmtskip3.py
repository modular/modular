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
#   Path:   tests/data/simple_cases/fmtskip3.py
#
# ===----------------------------------------------------------------------=== #

a = 3
# fmt: off
b,    c = 1, 2
d =    6  # fmt: skip
e = 5
# fmt: on
f = [
    "This is a very long line that should be formatted into a clearer line ",
    "by rearranging.",
]

# output

a = 3
# fmt: off
b,    c = 1, 2
d =    6  # fmt: skip
e = 5
# fmt: on
f = [
    "This is a very long line that should be formatted into a clearer line ",
    "by rearranging.",
]
