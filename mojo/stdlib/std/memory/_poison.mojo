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
"""Debug allocator poison pattern detection for uninitialized memory reads.

When compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`, the helpers in
this module check loaded float values against the bit pattern written by
the debug allocator (`MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison`).
A match triggers `abort()` with a descriptive message. When disabled (the
default), `@parameter if` / `comptime if` eliminates all checking code at
compile time with zero runtime overhead.

The poison pattern is the bit pattern of the largest finite value of the
float type (e.g. `FLT_MAX` for `float32`). This is intentionally non-NaN
and non-Inf so the poison fill does not conflict with the `nan-check`
debug pass when kernels leave allocation padding un-overwritten.
"""

from std.builtin.dtype import _unsigned_integral_type_of
from std.os import abort
from std.sys.defines import get_defined_string
from std.utils.numerics import max_finite

comptime _UNINIT_CHECK_ENABLED = (
    get_defined_string["MOJO_STDLIB_SIMD_UNINIT_CHECK", "false"]() == "true"
)


# Float dtypes that participate in the poison check. The set must stay in
# sync with the dtypes that `Device::createDeviceMemory` poison-fills in
# `Driver.cpp`. Excluded:
# - `float4_e2m1fn`: sub-byte; the byte-granularity allocator fill can't
#   write a single 4-bit lane, so the C++ side skips the fill.
# - `float8_e3m4`, `float8_e8m0fnu`: `max_finite` is not defined for these
#   scale-only / no-mantissa formats, so the C++ side skips the fill too.
@always_inline
def _is_poison_checked_dtype[dtype: DType]() -> Bool:
    return (
        dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.float8_e4m3fn
        or dtype == DType.float8_e4m3fnuz
        or dtype == DType.float8_e5m2
        or dtype == DType.float8_e5m2fnuz
    )


@always_inline
def _check_not_poison[dtype: DType, width: Int](val: SIMD[dtype, width]):
    """Checks that a loaded SIMD value doesn't match debug allocator poison.

    Only active when compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`. Zero cost otherwise.
    """

    comptime if not _UNINIT_CHECK_ENABLED:
        return

    comptime if _is_poison_checked_dtype[dtype]():
        comptime uint_type = _unsigned_integral_type_of[dtype]()
        comptime poison = max_finite[dtype]().to_bits[uint_type]()

        var bits = val.to_bits[uint_type]()
        var any_poisoned = False
        for i in range(width):
            if bits[i] == poison:
                any_poisoned = True
                break
        if any_poisoned:
            abort(
                "use of uninitialized memory: loaded value matches"
                " debug allocator poison pattern"
            )


@always_inline
def _check_not_poison_masked[
    dtype: DType, width: Int
](val: SIMD[dtype, width], mask: SIMD[DType.bool, width]):
    """Checks unmasked lanes of a SIMD value for debug allocator poison.

    Only active when compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`. Zero cost otherwise.
    Masked-off lanes contain passthrough values and are not checked.
    """
    comptime if not _UNINIT_CHECK_ENABLED:
        return

    comptime if _is_poison_checked_dtype[dtype]():
        # Work at the integer-bit level to avoid ARM NaN canonicalization.
        # Float-level select can convert arbitrary NaN to canonical qNaN,
        # causing false positives in masked-off lanes.
        comptime uint_type = _unsigned_integral_type_of[dtype]()
        comptime poison = max_finite[dtype]().to_bits[uint_type]()

        var bits = val.to_bits[uint_type]()
        var safe_bits = mask.select(bits, SIMD[uint_type, width](0))
        var any_poisoned = False
        for i in range(width):
            if safe_bits[i] == poison:
                any_poisoned = True
                break
        if any_poisoned:
            abort(
                "use of uninitialized memory: loaded value matches"
                " debug allocator poison pattern"
            )
