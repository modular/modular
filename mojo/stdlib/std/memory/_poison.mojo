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

When compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`, the helpers in this module check
loaded float values against the known poison bit patterns written by the
debug allocator (`MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison`).
A match triggers `abort()` with a descriptive message. When disabled
(the default), `@parameter if` / `comptime if` eliminates all checking
code at compile time with zero runtime overhead.
"""

from std.builtin.dtype import _unsigned_integral_type_of
from std.os import abort
from std.sys.defines import get_defined_string
from std.memory import UnsafePointer
from std.sys.info import size_of
from std.utils.numerics import FPUtils

comptime _UNINIT_CHECK_ENABLED = (
    get_defined_string["MOJO_STDLIB_SIMD_UNINIT_CHECK", "false"]() == "true"
)


@always_inline
def _make_poison_pattern[dtype: DType, byte_val: Byte]() -> Scalar[dtype]:
    """Creates a scalar with all bytes set to `byte_val`."""
    comptime byte_size = size_of[dtype]()
    var splatted = SIMD[DType.uint8, byte_size](byte_val)
    return UnsafePointer(to=splatted).bitcast[Scalar[dtype]]()[]


@always_inline
def _canonical_qnan_bits[
    dtype: DType
]() -> Scalar[_unsigned_integral_type_of[dtype]()]:
    """Returns the bit pattern for canonical quiet NaN of the given float type.

    Canonical qNaN has all exponent bits set and the quiet bit (MSB of
    mantissa) set. This is the pattern used by LLVM's APFloat::getQNaN,
    which the debug allocator uses for device poison.
    """
    comptime exp_w = FPUtils[dtype].exponent_width()
    comptime man_w = FPUtils[dtype].mantissa_width()
    # Exponent all-1s shifted to position, OR quiet bit (mantissa MSB).
    # Use explicit parentheses to avoid operator precedence issues.
    comptime exp_mask = ((1 << exp_w) - 1) << man_w
    comptime quiet_bit = 1 << (man_w - 1)
    comptime uint_type = _unsigned_integral_type_of[dtype]()
    return Scalar[uint_type](exp_mask | quiet_bit)


@always_inline
def _check_not_poison[dtype: DType, width: Int](val: SIMD[dtype, width]):
    """Checks that a loaded SIMD value doesn't match debug allocator poison.

    Only active when compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`. Zero cost otherwise.
    """

    comptime if not _UNINIT_CHECK_ENABLED:
        return

    comptime if dtype == DType.bool or dtype.is_integral():
        return

    comptime if dtype == DType.float8_e4m3fnuz or dtype == DType.float8_e5m2fnuz:
        return  # 0xFF is negative zero in fnuz types, can't use as poison

    comptime if dtype.is_floating_point():
        comptime uint_type = _unsigned_integral_type_of[dtype]()
        var bits = val.to_bits[uint_type]()

        # Host poison: all 0xFF bytes
        comptime host_poison = _make_poison_pattern[uint_type, 0xFF]()

        # Device poison: canonical qNaN
        comptime device_poison = _canonical_qnan_bits[dtype]()

        var any_poisoned = False
        for i in range(width):
            if bits[i] == host_poison or bits[i] == device_poison:
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

    comptime if not dtype.is_floating_point():
        return

    # Work at the integer-bit level to avoid ARM NaN canonicalization.
    # Float-level select can convert arbitrary NaN to canonical qNaN,
    # causing false positives in masked-off lanes.
    comptime uint_type = _unsigned_integral_type_of[dtype]()
    var bits = val.to_bits[uint_type]()
    var safe_bits = mask.select(bits, SIMD[uint_type, width](0))

    comptime host_poison = _make_poison_pattern[uint_type, 0xFF]()
    comptime device_poison = _canonical_qnan_bits[dtype]()

    var any_poisoned = False
    for i in range(width):
        if safe_bits[i] == host_poison or safe_bits[i] == device_poison:
            any_poisoned = True
            break
    if any_poisoned:
        abort(
            "use of uninitialized memory: loaded value matches"
            " debug allocator poison pattern"
        )
