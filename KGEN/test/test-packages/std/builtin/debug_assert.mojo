# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Minimal debug_assert stub for isolated parser tests."""


def debug_assert(cond: Bool):
    if not cond:
        abort()


def debug_assert(cond: Bool, message: StringLiteral):
    if not cond:
        abort()


@no_inline
def abort() -> Never:
    __mlir_op.`llvm.intr.trap`()
    while True:
        pass
