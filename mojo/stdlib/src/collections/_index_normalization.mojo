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
"""The utilities provided in this module help normalize the access
to data elements in arrays."""

from sys.intrinsics import _type_is_eq
from sys import sizeof


@always_inline
fn normalize_index[
    I: Indexer, //, container_name: StringLiteral
](idx: I, length: UInt) -> UInt:
    """Normalize the given index value to a valid index value for the given container length.

    If the provided value is negative, the `index + container_length` is returned.

    Parameters:
        I: A type that can be used as an index.
        container_name: The name of the container. Used for the error message.

    Args:
        idx: The index value to normalize.
        length: The container length to normalize the index for.

    Returns:
        The normalized index value.
    """

    @parameter
    if (
        _type_is_eq[I, UInt]()
        or _type_is_eq[I, UInt8]()
        or _type_is_eq[I, UInt16]()
        or _type_is_eq[I, UInt32]()
        or _type_is_eq[I, UInt64]()
    ):
        var i = UInt(index(idx))
        # TODO: Consider a way to construct the error message after the assert has failed
        # something like "Indexing into an empty container" if length == 0 else "..."
        debug_assert[assert_mode="safe", cpu_only=True](
            i < length,
            container_name,
            " index out of bounds: index (",
            i,
            ") valid range: -",  # can't print -UInt.MAX
            length,
            " <= index < ",
            length,
        )
        return i
    else:
        var mlir_index = index(idx)
        var i = UInt(mlir_index)
        if Int(mlir_index) < 0:
            i += length
        # Checking the bounds after the normalization saves a comparison
        # while allowing negative indexing into containers with length > Int.MAX.
        # For a positive index this is trivially correct.
        # For a negative index we can infer the full bounds check from
        # the assert UInt(idx + length) < length, by considering 2 cases:
        #   when length > Int.MAX then:
        #     idx + length > idx + Int.MAX >= Int.MIN + Int.MAX = -1
        #     therefore idx + length >= 0
        #   when length <= Int.MAX then:
        #     UInt(idx + length) < length <= Int.MAX
        #     Which means UInt(idx + length) signed bit is off
        #     therefore idx + length >= 0
        # in either case we can infer 0 <= idx + length < length
        debug_assert[assert_mode="safe", cpu_only=True](
            i < length,
            container_name,
            " index out of bounds: index (",
            Int(mlir_index),
            ") valid range: -",  # can't print -UInt.MAX
            length,
            " <= index < ",
            length,
        )
        return i


@always_inline
fn normalize_index[
    I: Indexer, //, container_name: StringLiteral
](idx: I, length: Int) -> Int:
    """Normalize the given index value to a valid index value for the given container length.

    If the provided value is negative, the `index + container_length` is returned.

    Parameters:
        I: A type that can be used as an index.
        container_name: The name of the container. Used for the error message.

    Args:
        idx: The index value to normalize.
        length: The container length to normalize the index for.

    Returns:
        The normalized index value.
    """
    return Int(normalize_index[container_name](idx, UInt(length)))
