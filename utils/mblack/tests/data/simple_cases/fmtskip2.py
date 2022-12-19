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
#   Path:   tests/data/simple_cases/fmtskip2.py
#
# ===----------------------------------------------------------------------=== #

l1 = [
    "This list should be broken up",
    "into multiple lines",
    "because it is way too long",
]
l2 = ["But this list shouldn't", "even though it also has", "way too many characters in it"]  # fmt: skip
l3 = [
    "I have",
    "trailing comma",
    "so I should be braked",
]

# output

l1 = [
    "This list should be broken up",
    "into multiple lines",
    "because it is way too long",
]
l2 = ["But this list shouldn't", "even though it also has", "way too many characters in it"]  # fmt: skip
l3 = [
    "I have",
    "trailing comma",
    "so I should be braked",
]
