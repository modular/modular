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

"""Mojo kernel wrappers for matmul MO interpreter operations."""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator

from algorithm.functional import IndexList
from memory import OpaquePointer
from linalg.matmul import matmul
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout
from runtime.asyncrt import DeviceContextPtr

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx


fn _is_gpu_allowed_matmul_dtype[dtype: DType]() -> Bool:
    """Check if a dtype is allowed for GPU matmul at compile time.

    GPU matmul does not support int8, uint8, int16, uint16, or float64.
    """

    # TODO(MXF-109): Add support for other dtypes.
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
    )


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_matmul_ops() -> PythonObject:
    """Create a Python module with matmul kernel function bindings."""
    try:
        var b = PythonModuleBuilder("matmul_ops")

        b.def_function[matmul_dispatcher](
            "Matmul", docstring="Matrix multiplication"
        )

        return b.finalize()
    except e:
        abort(String("failed to create matmul op bindings module: ", e))


# =============================================================================
# Dispatcher
# =============================================================================


fn matmul_dispatcher(
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Matmul dispatcher with dtype dispatch.

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The left-hand side buffer object.
        rhs_buffer: The right-hand side buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(lhs_buffer)
    var rhs_dtype = _get_dtype(rhs_buffer)
    if dtype != rhs_dtype:
        raise Error(
            "Mismatched input dtypes for matmul: "
            + String(dtype)
            + " and "
            + String(rhs_dtype)
        )

    # Extract shapes: lhs is (M, K), rhs is (K, N), out is (M, N)
    var lhs_shape = lhs_buffer.shape
    var m = Int(py=lhs_shape[0])
    var k = Int(py=lhs_shape[1])
    var rhs_shape = rhs_buffer.shape
    var n = Int(py=rhs_shape[1])

    var ctx = _get_ctx(device_context_ptr)

    # Float types
    if dtype == DType.float16:
        matmul_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](lhs_buffer),
            _get_buffer_ptr[DType.float16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.float32:
        matmul_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](lhs_buffer),
            _get_buffer_ptr[DType.float32](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.float64:
        matmul_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](lhs_buffer),
            _get_buffer_ptr[DType.float64](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.bfloat16:
        matmul_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](lhs_buffer),
            _get_buffer_ptr[DType.bfloat16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    # Integer types
    elif dtype == DType.int8:
        matmul_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](lhs_buffer),
            _get_buffer_ptr[DType.int8](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.int16:
        matmul_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](lhs_buffer),
            _get_buffer_ptr[DType.int16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.int32:
        matmul_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](lhs_buffer),
            _get_buffer_ptr[DType.int32](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.int64:
        matmul_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](lhs_buffer),
            _get_buffer_ptr[DType.int64](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint8:
        matmul_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](lhs_buffer),
            _get_buffer_ptr[DType.uint8](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint16:
        matmul_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](lhs_buffer),
            _get_buffer_ptr[DType.uint16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint32:
        matmul_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](lhs_buffer),
            _get_buffer_ptr[DType.uint32](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint64:
        matmul_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](lhs_buffer),
            _get_buffer_ptr[DType.uint64](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for matmul: " + String(dtype))


# =============================================================================
# Kernel implementation
# =============================================================================


@always_inline
fn matmul_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int,
    k: Int,
    n: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Matrix multiplication: out = lhs @ rhs.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        lhs_ptr: Pointer to the left-hand side buffer data.
        rhs_ptr: Pointer to the right-hand side buffer data.
        m: Number of rows in lhs and output.
        k: Number of columns in lhs / rows in rhs.
        n: Number of columns in rhs and output.
        ctx: Device context pointer (null for CPU).
    """
    # Define static layout type with unknown dimensions for 2D row-major matrices
    comptime layout_2d = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime LayoutType = RuntimeLayout[layout_2d]

    # Create LayoutTensors with runtime shapes
    var c = LayoutTensor[dtype, layout_2d, MutExternalOrigin](
        out_ptr, LayoutType.row_major(IndexList[2](m, n))
    )
    var a = LayoutTensor[dtype, layout_2d, MutExternalOrigin](
        lhs_ptr, LayoutType.row_major(IndexList[2](m, k))
    )
    var b = LayoutTensor[dtype, layout_2d, MutExternalOrigin](
        rhs_ptr, LayoutType.row_major(IndexList[2](k, n))
    )

    if not ctx:
        # TODO(MXF-108): Remove single_thread_blocking_override
        matmul[target="cpu", single_thread_blocking_override=True](
            c, a, b, None
        )
    else:
        # GPU execution - check GPU availability and dtype support
        @parameter
        if has_accelerator():

            @parameter
            if _is_gpu_allowed_matmul_dtype[dtype]():
                var device_ctx = DeviceContextPtr(ctx)
                matmul[target="gpu"](c, a, b, device_ctx.get_device_context())
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for matmul with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")
