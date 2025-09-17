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
#   Path:   tests/data/simple_cases/fmtonoff4.py
#
# ===----------------------------------------------------------------------=== #

# fmt: off
@test([
    1, 2,
    3, 4,
])
# fmt: on
def f():
    pass


@test(
    [
        1,
        2,
        3,
        4,
    ]
)
def f():
    pass


# output

# fmt: off
@test([
    1, 2,
    3, 4,
])
# fmt: on
def f():
    pass


@test(
    [
        1,
        2,
        3,
        4,
    ]
)
def f():
    pass
