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
"""Provides low-level GPU intrinsic operations and memory access primitives.
"""


from std.sys import is_nvidia_gpu
from std.sys.intrinsics import llvm_intrinsic
from std.sys._assembly import inlined_assembly

# ===-----------------------------------------------------------------------===#
# mulhi
# ===-----------------------------------------------------------------------===#


@inline(.always)
def mulhi(a: UInt16, b: UInt16) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 16-bit
    unsigned integers.

    Multiplies two 16-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow
    detection.

    Args:
        a: First 16-bit unsigned integer operand.
        b: Second 16-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        This performs the multiplication using 32-bit arithmetic.
    """

    var au32 = a.cast[.uint32]()
    var bu32 = b.cast[.uint32]()
    return au32 * bu32


@inline(.always)
def mulhi(a: Int16, b: Int16) -> Int32:
    """Calculates the most significant 32 bits of the product of two 16-bit
    signed integers.

    Multiplies two 16-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 16-bit signed integer operand.
        b: Second 16-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        This performs the multiplication using 32-bit arithmetic.
    """

    var ai32 = a.cast[.int32]()
    var bi32 = b.cast[.int32]()
    return ai32 * bi32


@inline(.always)
def mulhi(a: UInt32, b: UInt32) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U32 PTX instruction.
        On others, it performs multiplication using 64-bit arithmetic.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.umulh", UInt32, has_side_effect=False](a, b)

    var au64 = a.cast[.uint64]()
    var bu64 = b.cast[.uint64]()
    return ((au64 * bu64) >> 32).cast[.uint32]()


@inline(.always)
def mulhi(a: Int32, b: Int32) -> Int32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    signed integers.

    Multiplies two 32-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S32 PTX instruction.
        On others, it performs multiplication using 64-bit arithmetic.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.smulh", Int32, has_side_effect=False](a, b)

    var ai64 = a.cast[.int64]()
    var bi64 = b.cast[.int64]()
    return ((ai64 * bi64) >> 32).cast[.int32]()


@inline(.always)
def mulhi(a: UInt64, b: UInt64) -> UInt64:
    """Calculates the most significant 64 bits of the product of two 64-bit
    unsigned integers.

    Multiplies two 64-bit unsigned integers and returns the high 64 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 64-bit unsigned integer operand.
        b: Second 64-bit unsigned integer operand.

    Returns:
        The high 64 bits of the product a * b.

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U64 PTX instruction.
        On others, it performs multiplication using 128-bit arithmetic.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.umulh", UInt64, has_side_effect=False](a, b)

    var au128 = a.cast[.uint128]()
    var bu128 = b.cast[.uint128]()
    return ((au128 * bu128) >> 64).cast[.uint64]()


@inline(.always)
def mulhi(a: Int64, b: Int64) -> Int64:
    """Calculates the most significant 64 bits of the product of two 64-bit
    signed integers.

    Multiplies two 64-bit signed integers and returns the high 64 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 64-bit signed integer operand.
        b: Second 64-bit signed integer operand.

    Returns:
        The high 64 bits of the product a * b.

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S64 PTX instruction.
        On others, it performs multiplication using 128-bit arithmetic.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.smulh", Int64, has_side_effect=False](a, b)

    var ai128 = a.cast[.int128]()
    var bi128 = b.cast[.int128]()
    return ((ai128 * bi128) >> 64).cast[.int64]()


# ===-----------------------------------------------------------------------===#
# mulwide
# ===-----------------------------------------------------------------------===#


@inline(.always)
def mulwide(a: UInt32, b: UInt32) -> UInt64:
    """Performs a wide multiplication of two 32-bit unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The full 64-bit product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.U32 PTX instruction.
        On others, it performs multiplication using 64-bit casts.
    """

    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.wide.u32 $0, $1, $2;",
            UInt64,
            constraints="=l,r,r",
            has_side_effect=False,
        ](a, b)

    var au64 = a.cast[.uint64]()
    var bu64 = b.cast[.uint64]()
    return au64 * bu64


@inline(.always)
def mulwide(a: Int32, b: Int32) -> Int64:
    """Performs a wide multiplication of two 32-bit signed integers.

    Multiplies two 32-bit signed integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits or be negative.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The full 64-bit signed product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.S32 PTX instruction.
        On others, it performs multiplication using 64-bit casts.
    """

    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.wide.s32 $0, $1, $2;",
            Int64,
            constraints="=l,r,r",
            has_side_effect=False,
        ](a, b)

    var ai64 = a.cast[.int64]()
    var bi64 = b.cast[.int64]()
    return ai64 * bi64
