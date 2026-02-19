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

"""Mojo kernel wrappers for broadcast MO interpreter operations."""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator

from algorithm.functional import IndexList
from memory import OpaquePointer
from runtime.asyncrt import DeviceContextPtr
from tensor.managed_tensor_slice import ManagedTensorSlice
from tensor.io_spec import Input, Output
from compiler_internal import StaticTensorSpec
from MOGGKernelAPI.MOGGKernelAPI import StaticBroadcastTo

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_broadcast_ops() -> PythonObject:
    """Create a Python module with broadcast kernel function bindings."""
    try:
        var b = PythonModuleBuilder("broadcast_ops")

        b.def_function[static_broadcast_to_dispatcher](
            "StaticBroadcastTo", docstring="Static broadcast to"
        )

        return b.finalize()
    except e:
        abort(String("failed to create broadcast op bindings module: ", e))


# =============================================================================
# Helpers
# =============================================================================


fn _pad_shape_to_rank5(
    shape_obj: PythonObject, rank: Int
) raises -> IndexList[MAX_RANK]:
    """Pad shape with leading 1s to make it rank-5."""
    var padded = IndexList[MAX_RANK]()
    var pad_count = MAX_RANK - rank
    for i in range(pad_count):
        padded[i] = 1
    for i in range(rank):
        padded[pad_count + i] = Int(py=shape_obj[i])
    return padded


# =============================================================================
# Kernel implementation
# =============================================================================


@always_inline
fn static_broadcast_to_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: IndexList[MAX_RANK],
    out_shape: IndexList[MAX_RANK],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Call StaticBroadcastTo.execute with rank-5 tensors.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        in_shape: Padded input shape (rank-5).
        out_shape: Padded output shape (rank-5).
        ctx: Device context pointer (null for CPU).
    """
    # Create ManagedTensorSlice wrappers
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )

    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl
        StaticBroadcastTo.execute[
            target="cpu",
            dtype=dtype,
            in_rank=MAX_RANK,
            out_rank=MAX_RANK,
            _trace_name="interpreter.static_broadcast_to",
            use_blocking_impl=True,
        ](output_tensor, input_tensor, out_shape, DeviceContextPtr())
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                StaticBroadcastTo.execute[
                    target="gpu",
                    dtype=dtype,
                    in_rank=MAX_RANK,
                    out_rank=MAX_RANK,
                    _trace_name="interpreter.static_broadcast_to",
                ](output_tensor, input_tensor, out_shape, device_ctx)
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for static_broadcast_to"
                    " with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# =============================================================================
# Dispatcher
# =============================================================================


fn static_broadcast_to_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    out_shape_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """StaticBroadcastTo dispatcher - unwraps PythonObjects and dispatches.

    Pads shapes to rank-5 with leading 1s and dispatches a single rank-5
    broadcast operation.
    """
    # Unwrap all PythonObjects upfront
    var dtype = _get_dtype(in_buffer)
    var in_shape_obj = in_buffer.shape
    var in_rank = Int(py=len(in_shape_obj))
    var out_rank = Int(py=len(out_shape_obj))
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    # Validate ranks
    if in_rank > MAX_RANK or out_rank > MAX_RANK:
        raise Error(
            "Unsupported rank: in_rank="
            + String(in_rank)
            + ", out_rank="
            + String(out_rank)
            + ". Max supported rank is "
            + String(MAX_RANK)
        )

    # Pad shapes to rank-5 with leading 1s
    var padded_in_shape = _pad_shape_to_rank5(in_shape_obj, in_rank)
    var padded_out_shape = _pad_shape_to_rank5(out_shape_obj, out_rank)

    @always_inline
    fn _make_ptr[
        dtype: DType
    ](addr: Int) -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
        return UnsafePointer[Scalar[dtype], MutExternalOrigin](
            unsafe_from_address=addr
        )

    # Dispatch by dtype
    if dtype == DType.float32:
        static_broadcast_to_op[DType.float32](
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.float64:
        static_broadcast_to_op[DType.float64](
            _make_ptr[DType.float64](out_addr),
            _make_ptr[DType.float64](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.float16:
        static_broadcast_to_op[DType.float16](
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.bfloat16:
        static_broadcast_to_op[DType.bfloat16](
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int8:
        static_broadcast_to_op[DType.int8](
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int16:
        static_broadcast_to_op[DType.int16](
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int32:
        static_broadcast_to_op[DType.int32](
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int64:
        static_broadcast_to_op[DType.int64](
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint8:
        static_broadcast_to_op[DType.uint8](
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint16:
        static_broadcast_to_op[DType.uint16](
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint32:
        static_broadcast_to_op[DType.uint32](
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint64:
        static_broadcast_to_op[DType.uint64](
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.bool:
        static_broadcast_to_op[DType.bool](
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    else:
        raise Error(
            "Unsupported dtype for static_broadcast_to: " + String(dtype)
        )
