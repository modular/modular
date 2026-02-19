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

"""Shared utilities for MO interpreter Mojo kernel wrappers."""

from python import PythonObject
from memory import OpaquePointer
from algorithm.functional import IndexList


comptime MAX_RANK = 5
"""The maximum rank of a tensor supported by the MO interpreter."""


fn _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


# Helper to extract buffer pointer with dtype
fn _get_buffer_ptr[
    dtype: DType
](buffer: PythonObject) raises -> UnsafePointer[
    Scalar[dtype], MutExternalOrigin
]:
    return UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=buffer._data_ptr())
    )


fn _get_size(buffer: PythonObject) raises -> Int:
    return Int(py=buffer.num_elements)


fn _get_ctx(
    device_context_ptr: PythonObject,
) raises -> OpaquePointer[MutExternalOrigin]:
    return OpaquePointer[MutExternalOrigin](
        unsafe_from_address=Int(py=device_context_ptr)
    )


fn _get_shape(
    shape_obj: PythonObject, rank: Int
) raises -> InlineArray[Int, MAX_RANK]:
    """Extract shape as InlineArray from Python sequence.

    Args:
        shape_obj: Python sequence containing the shape.
        rank: The rank of the shape.

    Returns:
        The shape as an InlineArray (only first `rank` elements are valid).
    """
    if rank > MAX_RANK:
        raise Error(
            "Tensor rank "
            + String(rank)
            + " exceeds MAX_RANK "
            + String(MAX_RANK)
        )
    var result = InlineArray[Int, MAX_RANK](fill=0)
    for i in range(rank):
        result[i] = Int(py=shape_obj[i])
    return result^
