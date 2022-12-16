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
#   Path:   tests/data/py_310/starred_for_target.py
#
# ===----------------------------------------------------------------------=== #

for x in *a, *b:
    print(x)

for x in a, b, *c:
    print(x)

for x in *a, b, c:
    print(x)

for x in *a, b, *c:
    print(x)

async for x in *a, *b:
    print(x)

async for x in *a, b, *c:
    print(x)

async for x in a, b, *c:
    print(x)

async for x in (
    *loooooooooooooooooooooong,
    very,
    *loooooooooooooooooooooooooooooooooooooooooooooooong,
):
    print(x)
