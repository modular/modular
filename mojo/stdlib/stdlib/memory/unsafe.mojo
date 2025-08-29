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
"""Provides utility functions for unsafe manipulation of SIMD values.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import bitcast
```
"""

from sys import bit_width_of
from sys import (
    is_amd_gpu,
    is_nvidia_gpu,
)

# ===-----------------------------------------------------------------------===#
# bitcast
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bitcast[
    src_dtype: DType,
    src_width: Int, //,
    dtype: DType,
    width: Int = src_width,
](val: SIMD[src_dtype, src_width]) -> SIMD[dtype, width]:
    """Bitcasts a SIMD value to another SIMD value.

    For a discussion of byte order, see
    [Converting data: bitcasting and byte order](/mojo/manual/pointers/unsafe-pointers#converting-data-bitcasting-and-byte-order)
    in the Mojo Manual.

    Examples:

    The following example uses `bitcast` to break a 32-bit integer into a vector
    of four 8-bit integers:

    ```mojo
    from memory import bitcast

    u32 = SIMD[DType.uint32, 1](4631)
    u8x4 = bitcast[DType.uint8, 4](u32)
    print(u32, u8x4) # 4631 [23, 18, 0, 0]
    ```

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        src_dtype: The source type.
        src_width: The source width.
        dtype: The target type.
        width: The target width.

    Args:
        val: The source value.

    Returns:
        A new SIMD value with the specified type and width with a bitcopy of the
        source SIMD value.
    """
    constrained[
        bit_width_of[SIMD[dtype, width]]() == bit_width_of[__type_of(val)](),
        "the source and destination types must have the same bitwidth",
    ]()

    # TODO(MOCO-2179): Change this to be more precise check for Arm devices, or
    # generate different ops on Arm.
    @parameter
    if not is_nvidia_gpu() and not is_amd_gpu():
        # Arm doesnt support casting between float16 and two ints.
        constrained[
            not (
                src_dtype == DType.float16
                and src_width == 1
                and dtype == DType.int8
                and width == 2
            ),
            "Can't cast a float16 directly to a 2 x i8",
        ]()
        constrained[
            not (
                src_dtype == DType.float16
                and src_width == 1
                and dtype == DType.uint8
                and width == 2
            ),
            "Can't cast a float16 directly to a 2 x ui8",
        ]()
        constrained[
            not (
                src_dtype == DType.int8
                and src_width == 2
                and dtype == DType.float16
                and width == 1
            ),
            "Can't cast a 2 x i8 directly to a float16",
        ]()
        constrained[
            not (
                src_dtype == DType.uint8
                and src_width == 2
                and dtype == DType.float16
                and width == 1
            ),
            "Can't cast a 2 x ui8 directly to a float16",
        ]()

    @parameter
    if dtype == src_dtype:
        return val._refine[dtype, width]()
    var res = __mlir_op.`pop.bitcast`[_type = SIMD[dtype, width]._mlir_type](
        val._mlir_value
    )
    return SIMD(mlir_value=res)


@always_inline("builtin")
fn _uint(n: Int) -> DType:
    # fmt: off
    return (
        DType._uint1 if n == 1 else
        DType._uint2 if n == 2 else
        DType._uint4 if n == 4 else
        DType.uint8 if n == 8 else
        DType.uint16 if n == 16 else
        DType.uint32 if n == 32 else
        DType.uint64 if n == 64 else
        DType.uint128 if n == 128 else
        DType.uint256 if n == 256 else
        DType.invalid
    )
    # fmt: on


fn _llvm_bitwidth(dtype: DType) -> Int:
    # fmt: off
    return (
        1 if dtype is DType._uint1 else
        2 if dtype is DType._uint2 else
        4 if dtype is DType._uint4 else
        8 if dtype is DType.uint8 else
        16 if dtype is DType.uint16 else
        32 if dtype is DType.uint32 else
        64 if dtype is DType.uint64 else
        128 if dtype is DType.uint128 else
        256 if dtype is DType.uint256 else
        -1
    )
    # fmt: on


@always_inline("nodebug")
fn pack_bits[
    src_width: Int, //,
    dtype: DType = _uint(src_width),
    width: Int = 1,
](val: SIMD[DType.bool, src_width]) -> SIMD[dtype, width]:
    """Packs a SIMD vector of `bool` values into an integer.

    Examples:

    This example packs a vector of 8 `bool` values into a single 8-bit integer.

    ```mojo
    from memory import pack_bits

    bits = SIMD[DType.bool, 8](1, 1, 0, 1, 0, 0, 0, 0)
    u8 = pack_bits[DType.uint8](bits)
    print(bits, u8) # [True, True, False, True, False, False, False, False] 11
    ```

    Constraints:
        The logical bitwidth of the bool vector must be the same as the bitwidth of the
        target type. The target type must be a unsigned type.

    Parameters:
        src_width: The source width.
        dtype: The target type.
        width: The target width.

    Args:
        val: The source value.

    Returns:
        A new integer scalar which has the same bitwidth as the bool vector.
    """
    constrained[
        dtype.is_unsigned() and _llvm_bitwidth(dtype) * width == src_width,
        (
            "the logical bitwidth of the bool vector must be the same as the"
            " target type"
        ),
    ]()

    var res = __mlir_op.`pop.bitcast`[_type = SIMD[dtype, width]._mlir_type](
        val._mlir_value
    )
    return SIMD(mlir_value=res)
