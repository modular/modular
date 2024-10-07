# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

# https://github.com/PaddlePaddle/Paddle/blob/3862f8303d2723c03ffb42ce332d4c570906669f/paddle/phi/kernels/funcs/weight_only_gemv.cu#L795

# logic and shift instruciton: lop3
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3

from memory.unsafe import bitcast
from utils import StaticTuple
from sys import has_neon
from testing import assert_equal


# 8xint4 -> 8xbfloat16 interleaved conversion
fn int4tobf16(i4: Int32) -> SIMD[DType.bfloat16, 8]:
    alias MASK: Int32 = 0x000F000F
    alias I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300

    var st: StaticTuple[SIMD[DType.bfloat16, 2], 4] = 0
    var i4s: Int32 = i4
    # Emulate the lop3 instruction
    # In this case the operation is: (A & B) | C
    st[0] = bitcast[DType.bfloat16, 2](((i4s & MASK) | I4s_TO_BF16s_MAGIC_NUM))

    @parameter
    for i in range(1, 4):
        i4s /= 16
        st[i] = bitcast[DType.bfloat16, 2](
            (i4s & MASK) | I4s_TO_BF16s_MAGIC_NUM
        )

    # 0xc308 = -136.0, 0xc300 = -128.0
    alias BF16_BIAS = SIMD[DType.bfloat16, 2](-128, -128)
    # 0x3f80 = 1.0
    alias BF16_ONE = SIMD[DType.bfloat16, 2](1, 1)

    @parameter
    for i in range(4):
        st[i] = st[i].fma(BF16_ONE, BF16_BIAS)

    var bf16v: SIMD[DType.bfloat16, 8] = 0

    @parameter
    for i in range(4):
        bf16v[2 * i + 0] = st[i][0]
        bf16v[2 * i + 1] = st[i][1]

    return bf16v


def test_int4tobfloat16():
    var bf16v: SIMD[DType.bfloat16, 8] = 0
    for i in range(4):
        bf16v[2 * i + 0] = i + 0
        bf16v[2 * i + 1] = i + 4
    assert_equal(int4tobf16(0x76543210), bf16v)


def main():
    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        test_int4tobfloat16()
