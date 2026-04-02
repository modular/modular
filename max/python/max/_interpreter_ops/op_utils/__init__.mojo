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

from std.python import PythonObject
from std.memory import OpaquePointer
from std.algorithm.functional import IndexList
from std.sys.info import has_apple_gpu_accelerator


comptime MAX_RANK = 5
"""The maximum rank of a tensor supported by the MO interpreter."""


def _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


# Helper to extract buffer pointer with dtype
def _get_buffer_ptr[
    dtype: DType
](buffer: PythonObject) raises -> UnsafePointer[
    Scalar[dtype], MutExternalOrigin
]:
    return UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=buffer._data_ptr())
    )


def _get_size(buffer: PythonObject) raises -> Int:
    return Int(py=buffer.num_elements)


@always_inline
def _make_ptr[
    dtype: DType
](addr: Int) -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
    """Create a typed pointer from a raw integer address."""
    return UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=addr
    )


def _get_ctx(
    device_context_ptr: PythonObject,
) raises -> OpaquePointer[MutExternalOrigin]:
    return OpaquePointer[MutExternalOrigin](
        unsafe_from_address=Int(py=device_context_ptr)
    )


trait Dispatchable:
    """Trait for operations that can be dispatched over all data dtypes.

    Implement `call[t: DType](self)` to perform the operation for a specific
    data dtype. Use `dispatch_dtype` to invoke the correct specialization at
    runtime based on a `DType` value.
    """

    def call[t: DType](self) raises -> None:
        """Perform the operation for the given compile-time data dtype.

        Parameters:
            t: The data dtype to specialize for.

        Raises:
            Error: If the operation fails for the given dtype.
        """
        ...


def dispatch_dtype[T: Dispatchable](body: T, dtype: DType) raises:
    """Dispatch to `body.call[dtype]()` for supported data dtypes.
    Useful when using _get_dtype/dtype is only runtime-known.

    Parameters:
        T: A type implementing `Dispatchable`.

    Args:
        body: The operation body to dispatch.
        dtype: The runtime dtype to dispatch on.

    Raises:
        Error: If `dtype` is not a supported data dtype.
    """
    if dtype == DType.float32:
        body.call[DType.float32]()
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            body.call[DType.float64]()
        else:
            raise Error("float64 is not supported on Apple GPU")
    elif dtype == DType.float16:
        body.call[DType.float16]()
    elif dtype == DType.bfloat16:
        body.call[DType.bfloat16]()
    elif dtype == DType.int8:
        body.call[DType.int8]()
    elif dtype == DType.int16:
        body.call[DType.int16]()
    elif dtype == DType.int32:
        body.call[DType.int32]()
    elif dtype == DType.int64:
        body.call[DType.int64]()
    elif dtype == DType.uint8:
        body.call[DType.uint8]()
    elif dtype == DType.uint16:
        body.call[DType.uint16]()
    elif dtype == DType.uint32:
        body.call[DType.uint32]()
    elif dtype == DType.uint64:
        body.call[DType.uint64]()
    elif dtype == DType.bool:
        body.call[DType.bool]()
    else:
        raise Error("Unsupported dtype: " + String(dtype))


def _get_shape(
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
