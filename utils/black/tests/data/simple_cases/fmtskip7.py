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
#   Path:   tests/data/simple_cases/fmtskip7.py
#
# ===----------------------------------------------------------------------=== #

a =     "this is some code"
b =     5  #fmt:skip
c = 9  #fmt: skip
d = "thisisasuperlongstringthisisasuperlongstringthisisasuperlongstringthisisasuperlongstring"  #fmt:skip

# output

a = "this is some code"
b =     5  # fmt:skip
c = 9  # fmt: skip
d = "thisisasuperlongstringthisisasuperlongstringthisisasuperlongstringthisisasuperlongstring"  # fmt:skip
