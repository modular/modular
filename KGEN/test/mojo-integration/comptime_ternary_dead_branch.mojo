# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

# Regression test for MOCO-3577: when a comptime ternary condition evaluates to
# a constant bool only during elaboration, the dead branch must not be
# elaborated.  Previously, the dead branch was elaborated and any `comptime
# assert False` inside it would cause a spurious compilation failure.
#
# `#kgen.param.expr<current_target>` is symbolic at IR emission time and only
# resolves to a concrete target during elaboration, so any condition that
# references it cannot be folded by the parser's deadCodeCheck.


def simd_bit_width_direct() -> Int:
    # Reads simd_bit_width from the current target using kgen.target directly,
    # without importing anything from the standard library.
    # #kgen.param.expr<current_target> is non-concrete at IR emission time and
    # only resolves during elaboration, so the returned Int holds a non-concrete
    # param.expr.  simd_bit_width is always >= 8 on any supported target.
    comptime target = __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "simd_bit_width" : !kgen.string`,
            `> : index`,
        ]
    )


def failure() -> String:
    comptime assert False, "This code should not be compiled or executed"


def failure2(s: String) -> String:
    comptime assert False, "This code should not be compiled or executed"


def main():
    # The else branch (calling `failure()`) must be recognized as dead during
    # elaboration and not elaborated.
    # CHECK: abc
    var s = "abc" if comptime (simd_bit_width_direct() > 0) else failure()
    print(s)

    # Variant: the then-branch contains two chained paramOps — `failure2(failure())`.
    # The condition is always false (simd_bit_width is never negative), so the
    # then-branch is dead and its contents is not elaborated.
    # CHECK: def
    var t = failure2(failure()) if comptime (
        simd_bit_width_direct() < 0
    ) else "def"
    print(t)
