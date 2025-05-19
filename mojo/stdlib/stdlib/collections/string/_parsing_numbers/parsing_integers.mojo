# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from memory import memcmp, memcpy

from .constants import CONTAINER_SIZE, MAXIMUM_UINT64_AS_STRING


fn standardize_string_slice(
    x: StringSlice[mut=False],
) -> InlineArray[Byte, CONTAINER_SIZE]:
    """Put the input string in an inline array, aligned to the right and padded
    with "0" on the left.
    """
    var standardized_x = InlineArray[Byte, CONTAINER_SIZE](ord("0"))
    var std_x_ptr = standardized_x.unsafe_ptr()
    var x_len = x.byte_length()
    memcpy(std_x_ptr + CONTAINER_SIZE - x_len, x.unsafe_ptr(), x_len)
    return standardized_x


# The idea is to end up with a InlineArray of size
# 24, which is enough to store the largest integer
# that can be represented in unsigned 64 bits (size 20), and
# is also SIMD friendly because divisible by 8, 4, 2, 1.
# This 24 could be computed at compile time and adapted
# to the simd width and the base, but Mojo's compile-time
# computation is not yet powerful enough yet.
# For now we focus on base 10.
fn to_integer(x: StringSlice[mut=False]) raises -> UInt64:
    """The input does not need to be padded with "0" on the left.
    The function returns the integer value represented by the input string.
    """
    if x.byte_length() > MAXIMUM_UINT64_AS_STRING.byte_length():
        raise Error("The string has too many bytes: '", x.byte_length(), "'.")
    return to_integer(standardize_string_slice(x))


fn to_integer(
    standardized_x: InlineArray[Byte, CONTAINER_SIZE]
) raises -> UInt64:
    """Takes a inline array containing the ASCII representation of a number.

    Notes:
        It must be padded with "0" on the left. Using an InlineArray makes
        this SIMD friendly.

        We assume there are no leading or trailing whitespaces, no sign, no
        underscore.

        The function returns the integer value represented by the input string
        `"000000000048642165487456"` -> `48642165487456`.
    """

    var std_x_ptr = standardized_x.unsafe_ptr()
    # This could be done with simd if we see it's a bottleneck.
    for i in range(CONTAINER_SIZE):
        if not (Byte(ord("0")) <= std_x_ptr[i] <= Byte(ord("9"))):
            var num_str = StringSlice(
                ptr=std_x_ptr, length=len(standardized_x)
            ).lstrip("0")
            raise Error(
                "Invalid character(s) in the number: '",
                num_str,
                "' at index: ",
                i,
            )

    # 24 is not divisible by 16, so we stop at 8. Later on,
    # when we have better compile-time computation, we can
    # change 24 to be adapted to the simd width.
    alias simd_width = min(sys.simdwidthof[DType.uint64](), 8)

    var accumulator = SIMD[DType.uint64, simd_width](0)

    # We use memcmp to check that the number is not too large.
    alias max_standardized_x = String(UInt64.MAX).rjust(CONTAINER_SIZE, "0")
    var too_large = memcmp(
        std_x_ptr, max_standardized_x.unsafe_ptr(), CONTAINER_SIZE
    ) == 1
    if too_large:
        var num_str = StringSlice(
            ptr=std_x_ptr, length=len(standardized_x)
        ).lstrip("0")
        raise Error(
            "The string is too large to be converted to an integer: '",
            num_str,
            "'.",
        )

    # actual conversion
    alias vector_with_exponents = get_vector_with_exponents()

    @parameter
    for i in range(CONTAINER_SIZE // simd_width):
        var ascii_vector = (std_x_ptr + i * simd_width).load[width=simd_width]()
        var as_digits = ascii_vector - SIMD[DType.uint8, simd_width](ord("0"))
        var as_digits_index = as_digits.cast[DType.uint64]()
        alias vector_slice = (
            vector_with_exponents.unsafe_ptr() + i * simd_width
        ).load[width=simd_width]()
        accumulator += as_digits_index * vector_slice
    return Int(accumulator.reduce_add())


fn get_vector_with_exponents() -> InlineArray[UInt64, CONTAINER_SIZE]:
    """Returns (0, 0, 0, 0, 10**19, 10**18, 10**17, ..., 10, 1)."""
    var result = InlineArray[UInt64, CONTAINER_SIZE](0)
    for i in range(4, CONTAINER_SIZE):
        result[i] = 10 ** (CONTAINER_SIZE - i - 1)
    return result
