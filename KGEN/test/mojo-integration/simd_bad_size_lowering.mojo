# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Reject SIMD types whose resolved length is not within the acceptable range.
# FIXME: These should be correct on construction from the Mojo code itself,
# but as per MOCO-2839 we're (often) silently dropping assertions during the
# folding of @always_inline("builtin") functions. This is a clumsy fallback
# but a good backstop to ensure we don't generate invalid code.


def main():
    # CHECK: error: SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<0, f32>'
    var x = SIMD[DType.float32, 0](0)
    print(x)
    # CHECK: error: SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<-1, f32>'
    var y = SIMD[DType.float32, -1](0)
    print(y)
    # CHECK: error: SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<3, f32>'
    var z = SIMD[DType.float32, 3](0)
    print(z)
