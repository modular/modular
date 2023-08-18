# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo -debug-level full %s | FileCheck %s

# Evaluates the exp function using 6th order taylor series expansion. This is
# the same expansion used by MLAS internally. The expansion is:
#
# Exp[x] = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720 + x^7/5040
#        = 1 + x (1 + x (1/2 + x (1/6 + x (1/24 + (1/120 + x/720) x))))
fn exp_scalar_taylor_float32(x: Float32) -> Float32:
    return 1.0 + x * (
        1.0
        + x
        * (
            0.5
            + x
            * (0.166667 + x * (0.0416667 + (0.00833333 + 0.00138889 * x) * x))
        )
    )


fn main():
    let res = exp_scalar_taylor_float32(2.3)
    # CHECK: 9.88
    print(res)
