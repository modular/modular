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
#   Path:   tests/data/py_310/parenthesized_context_managers.py
#
# ===----------------------------------------------------------------------=== #

with (CtxManager() as example):
    ...

with (CtxManager1(), CtxManager2()):
    ...

with (CtxManager1() as example, CtxManager2()):
    ...

with (CtxManager1(), CtxManager2() as example):
    ...

with (CtxManager1() as example1, CtxManager2() as example2):
    ...

with (
    CtxManager1() as example1,
    CtxManager2() as example2,
    CtxManager3() as example3,
):
    ...
