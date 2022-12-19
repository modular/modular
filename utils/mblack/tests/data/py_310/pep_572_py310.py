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
#   Path:   tests/data/py_310/pep_572_py310.py
#
# ===----------------------------------------------------------------------=== #

# Unparenthesized walruses are now allowed in indices since Python 3.10.
x[a:=0]
x[a:=0, b:=1]
x[5, b:=0]

# Walruses are allowed inside generator expressions on function calls since 3.10.
if any(match := pattern_error.match(s) for s in buffer):
    if match.group(2) == data_not_available:
        # Error OK to ignore.
        pass

f(a := b + c for c in range(10))
f((a := b + c for c in range(10)), x)
f(y=(a := b + c for c in range(10)))
f(x, (a := b + c for c in range(10)), y=z, **q)
