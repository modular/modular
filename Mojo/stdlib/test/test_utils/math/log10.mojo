# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Provides inputs for the host and GPU `log10` tests."""

from std.math import inf, nan

from .rng import xorshift64

comptime num_log10_inputs = 16384
"""The number of values each `log10_*_inputs` function returns."""

comptime num_log10_specials = 11
"""The number of special values at the start of each `log10_*_inputs` list.

`10**n` follows them, at index `num_log10_specials + n`.
"""

comptime f32_min_subnormal_bits: UInt32 = 1
"""The bit pattern of the smallest positive float32 subnormal."""
comptime f32_max_subnormal_bits: UInt32 = 0x007F_FFFF
"""The bit pattern of the largest float32 subnormal."""
comptime f64_min_subnormal_bits: UInt64 = 1
"""The bit pattern of the smallest positive float64 subnormal."""
comptime f64_max_subnormal_bits: UInt64 = 0x000F_FFFF_FFFF_FFFF
"""The bit pattern of the largest float64 subnormal."""

comptime _sweep_seed: UInt64 = 0x2545F4914F6CDD1D


def log10_f64_inputs() -> List[Float64]:
    """Returns float64 inputs that exercise `log10`'s edge cases and range.

    Returns:
        A list of `num_log10_inputs` values.
    """
    # 2^64 / golden ratio, the Fibonacci-hashing multiplier.
    comptime golden_ratio_u64: UInt64 = 0x9E37_79B9_7F4A_7C15
    comptime f64_min_normal_bits: UInt64 = 0x0010_0000_0000_0000
    comptime f64_max_finite_bits: UInt64 = 0x7FEF_FFFF_FFFF_FFFF
    var v: List[Float64] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        inf[.float64](),
        -inf[.float64](),
        nan[.float64](),
        Float64(from_bits=f64_min_subnormal_bits),
        Float64(from_bits=f64_max_subnormal_bits),
        Float64(from_bits=f64_min_normal_bits),
        Float64(from_bits=f64_max_finite_bits),
    ]
    v.reserve(num_log10_inputs)
    var p: Float64 = 1.0
    for _ in range(309):
        v.append(p)
        p *= 10.0
    p = 0.1
    for _ in range(323):
        v.append(p)
        p /= 10.0
    # Subnormals: every bit position with a few mantissa tails.
    for i in range(52):
        var bit = UInt64(1) << UInt64(i)
        for j in range(4):
            v.append(
                Float64(
                    from_bits=bit | ((UInt64(j) * golden_ratio_u64) & (bit - 1))
                )
            )
    # Near 1, where log10 is small and cancellation matters.
    var one_bits = Float64(1.0).to_bits[.uint64]()
    for i in range(1, 400):
        var d = Float64(i) * 1.0e-6
        v.extend(
            [
                1.0 + d,
                1.0 - d,
                Float64(from_bits=one_bits + UInt64(i)),
                Float64(from_bits=one_bits - UInt64(i)),
            ]
        )

    for i in range(16):
        v.append(-Float64(i + 1) * 3.7)
    var state = _sweep_seed
    comptime normal_span = f64_max_finite_bits - f64_min_normal_bits
    while len(v) < num_log10_inputs:
        v.append(
            Float64(
                from_bits=f64_min_normal_bits + xorshift64(state) % normal_span
            )
        )
    return v^


def log10_f32_inputs() -> List[Float32]:
    """Returns float32 inputs that exercise `log10`'s edge cases and range.

    Returns:
        A list of `num_log10_inputs` values.
    """
    comptime f32_min_normal_bits: UInt32 = 0x0080_0000
    comptime f32_max_finite_bits: UInt32 = 0x7F7F_FFFF
    comptime f32_inf_bits = 0x7F80_0000
    var v: List[Float32] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        inf[.float32](),
        -inf[.float32](),
        nan[.float32](),
        Float32(from_bits=f32_min_subnormal_bits),
        Float32(from_bits=f32_max_subnormal_bits),
        Float32(from_bits=f32_min_normal_bits),
        Float32(from_bits=f32_max_finite_bits),
    ]
    v.reserve(num_log10_inputs)
    var p: Float32 = 1.0
    for _ in range(39):
        v.append(p)
        p *= 10.0
    # Near 1, where log10 is small and cancellation matters.
    var one_bits = Float32(1.0).to_bits[.uint32]()
    for i in range(1, 2048):
        var d = Float32(i) * 1.0e-4
        v.extend(
            [
                Float32(from_bits=one_bits + UInt32(i)),
                Float32(from_bits=one_bits - UInt32(i)),
                1.0 + d,
                1.0 - d,
            ]
        )
    for i in range(16):
        v.append(-Float32(i + 1) * 3.7)
    var state = _sweep_seed
    while len(v) < num_log10_inputs:
        var bits = (xorshift64(state) % f32_inf_bits).cast[.uint32]()
        v.append(Float32(from_bits=bits))
    return v^
