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

"""Mojo kernel wrappers for data movement MO interpreter operations.

Contains broadcast, transpose, memcpy, and slice operations.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, simd_width_of

from std.algorithm.functional import elementwise, IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice
from tensor.io_spec import Input, Output
from compiler_internal import StaticTensorSpec
from layout import IntTuple, create_unknown_int_tuple
from MOGGKernelAPI.MOGGKernelAPI import Slice, StaticBroadcastTo, Transpose
from op_utils import (
    _get_dtype,
    _get_buffer_ptr,
    _get_ctx,
    _get_size,
    _make_ptr,
    MAX_RANK,
    Dispatchable,
    dispatch_dtype,
)


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_data_movement_ops() -> PythonObject:
    """Create a Python module with data movement kernel function bindings."""
    try:
        var b = PythonModuleBuilder("data_movement_ops")

        b.def_function[static_broadcast_to_dispatcher](
            "StaticBroadcastTo", docstring="Static broadcast to"
        )
        b.def_function[transpose_dispatcher](
            "Transpose", docstring="Transpose operation"
        )
        b.def_function[memcpy_dispatcher](
            "Memcpy",
            docstring="Copy elements between buffers with offsets",
        )
        b.def_function[slice_dispatcher]("Slice", docstring="Slice operation")

        return b.finalize()
    except e:
        abort(t"failed to create data movement op bindings module: {e}")


# =============================================================================
# Helpers
# =============================================================================


def _pad_shape_to_max_rank(
    shape_obj: PythonObject, rank: Int
) raises -> IndexList[MAX_RANK]:
    """Pad shape with leading 1s to make it MAX_RANK."""
    var padded = IndexList[MAX_RANK]()
    var pad_count = MAX_RANK - rank
    for i in range(pad_count):
        padded[i] = 1
    for i in range(rank):
        padded[pad_count + i] = Int(py=shape_obj[i])
    return padded


# ===----------------------------------------------------------------------=== #
# Broadcast operation
# ===----------------------------------------------------------------------=== #


@always_inline
def static_broadcast_to_op[
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
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK, ...].get_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK, ...].get_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )

    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    if not ctx:
        StaticBroadcastTo.execute[
            target="cpu",
            dtype=dtype,
            in_rank=MAX_RANK,
            out_rank=MAX_RANK,
            _trace_name="interpreter.static_broadcast_to",
        ](output_tensor, input_tensor, out_shape, DeviceContextPtr())
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                StaticBroadcastTo.execute[
                    target="gpu",
                    dtype=dtype,
                    in_rank=MAX_RANK,
                    out_rank=MAX_RANK,
                    _trace_name="interpreter.static_broadcast_to",
                ](output_tensor, input_tensor, out_shape, device_ctx)
            else:
                raise Error(
                    "GPU execution not supported for static_broadcast_to"
                    " with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


@fieldwise_init
struct _StaticBroadcastToBody(Dispatchable):
    """Dispatch body for the StaticBroadcastTo operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var in_shape: IndexList[MAX_RANK]
    var out_shape: IndexList[MAX_RANK]
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        static_broadcast_to_op[t](
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.in_shape,
            self.out_shape,
            self.ctx,
        )


def static_broadcast_to_dispatcher(
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
    var padded_in_shape = _pad_shape_to_max_rank(in_shape_obj, in_rank)
    var padded_out_shape = _pad_shape_to_max_rank(out_shape_obj, out_rank)

    dispatch_dtype(
        _StaticBroadcastToBody(
            out_addr, in_addr, padded_in_shape, padded_out_shape, ctx
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# Transpose operation
# ===----------------------------------------------------------------------=== #


@always_inline
def transpose_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: IndexList[MAX_RANK],
    out_shape: IndexList[MAX_RANK],
    perm_data: InlineArray[Int64, MAX_RANK],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Call Transpose.execute with MAX_RANK tensors.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        in_shape: Padded input shape (MAX_RANK).
        out_shape: Padded output shape (MAX_RANK).
        perm_data: Padded permutation array (MAX_RANK).
        ctx: Device context pointer (null for CPU).
    """
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK, ...].get_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK, ...].get_unknown()
    comptime perm_spec = StaticTensorSpec[DType.int64, 1, ...].get_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )

    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    # TODO: ManagedTensorSlice should correctly propagate mutability to
    # prevent us from needing to unsafely cast the pointer mutability here.
    var perm_data_ptr = perm_data.unsafe_ptr().unsafe_mut_cast[True]()
    var perm_tensor = ManagedTensorSlice[io_spec=Input, static_spec=perm_spec](
        perm_data_ptr, IndexList[1](MAX_RANK)
    )

    if not ctx:
        Transpose.execute[
            target="cpu",
            _trace_name="interpreter.transpose",
            static_permutations=create_unknown_int_tuple(MAX_RANK),
            dtype=dtype,
            rank=MAX_RANK,
        ](output_tensor, input_tensor, perm_tensor, DeviceContextPtr())
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                Transpose.execute[
                    target="gpu",
                    _trace_name="interpreter.transpose",
                    static_permutations=create_unknown_int_tuple(MAX_RANK),
                    dtype=dtype,
                    rank=MAX_RANK,
                ](output_tensor, input_tensor, perm_tensor, device_ctx)
            else:
                raise Error(
                    "GPU execution not supported for transpose"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


@fieldwise_init
struct _TransposeBody(Dispatchable):
    """Dispatch body for the Transpose operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var in_shape: IndexList[MAX_RANK]
    var out_shape: IndexList[MAX_RANK]
    var perm: InlineArray[Int64, MAX_RANK]
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        transpose_op[t](
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.in_shape,
            self.out_shape,
            self.perm,
            self.ctx,
        )


def transpose_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    perm_obj: PythonObject,
    in_shape_obj: PythonObject,
    out_shape_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Transpose dispatcher - unwraps PythonObjects and dispatches.

    Pads shapes and permutation to MAX_RANK and dispatches.

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        perm_obj: Python list of permutation indices (original rank).
        in_shape_obj: Python sequence of input shape.
        out_shape_obj: Python sequence of output shape.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var in_rank = Int(py=len(in_shape_obj))
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    # Validate ranks
    if in_rank > MAX_RANK:
        raise Error(
            "Unsupported rank: "
            + String(in_rank)
            + ". Max supported rank is "
            + String(MAX_RANK)
        )

    # Pad shapes to MAX_RANK with leading 1s
    var padded_in_shape = _pad_shape_to_max_rank(in_shape_obj, in_rank)
    var padded_out_shape = _pad_shape_to_max_rank(out_shape_obj, in_rank)

    # Pad permutation: identity for leading dims, shifted original perm
    var pad_count = MAX_RANK - in_rank
    var padded_perm = InlineArray[Int64, MAX_RANK](fill=0)
    for i in range(pad_count):
        padded_perm[i] = Int64(i)
    for i in range(in_rank):
        padded_perm[pad_count + i] = Int64(Int(py=perm_obj[i]) + pad_count)

    dispatch_dtype(
        _TransposeBody(
            out_addr,
            in_addr,
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# Memcpy operation (copy elements between buffers with offsets)
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _MemcpyBody(Dispatchable):
    """Dispatch body for the Memcpy operation over data dtypes."""

    var dst_addr: Int
    var src_addr: Int
    var dst_offset: Int
    var src_offset: Int
    var count: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        memcpy_op[t](
            _make_ptr[t](self.dst_addr),
            _make_ptr[t](self.src_addr),
            self.dst_offset,
            self.src_offset,
            self.count,
            self.ctx,
        )


def memcpy_dispatcher(
    dst_buffer: PythonObject,
    src_buffer: PythonObject,
    dst_offset: PythonObject,
    src_offset: PythonObject,
    count: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Copy elements from src to dst buffer with offsets.

    Args:
        dst_buffer: The destination buffer object.
        src_buffer: The source buffer object.
        dst_offset: Element offset into the destination buffer.
        src_offset: Element offset into the source buffer.
        count: Number of elements to copy.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(src_buffer)
    var dst_dtype = _get_dtype(dst_buffer)
    if dtype != dst_dtype:
        raise Error(
            "Mismatched dtypes for memcpy: "
            + String(dtype)
            + " and "
            + String(dst_dtype)
        )

    var d_off = Int(py=dst_offset)
    var s_off = Int(py=src_offset)
    var cnt = Int(py=count)
    var ctx = _get_ctx(device_context_ptr)
    var dst_addr = Int(py=dst_buffer._data_ptr())
    var src_addr = Int(py=src_buffer._data_ptr())

    dispatch_dtype(
        _MemcpyBody(dst_addr, src_addr, d_off, s_off, cnt, ctx),
        dtype,
    )


@always_inline
def memcpy_op[
    dtype: DType
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dst_offset: Int,
    src_offset: Int,
    count: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Copy count elements from src+src_offset to dst+dst_offset.

    Parameters:
        dtype: The data type of the buffers.

    Args:
        dst_ptr: Pointer to the destination buffer data.
        src_ptr: Pointer to the source buffer data.
        dst_offset: Element offset into the destination.
        src_offset: Element offset into the source.
        count: Number of elements to copy.
        ctx: Device context pointer (null for CPU).
    """
    var d = dst_ptr + dst_offset
    var s = src_ptr + src_offset

    @always_inline
    @parameter
    @__copy_capture(d, s)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        d.store[width=width](i, s.load[width=width](i))

    if not ctx:
        elementwise[func, simd_width=simd_width_of[dtype]()](
            IndexList[1](count)
        )
    else:
        # GPU execution
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](count), device_ctx
                )
            else:
                raise Error(
                    "GPU execution not supported for memcpy with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


# ===----------------------------------------------------------------------=== #
# Slice operation
# ===----------------------------------------------------------------------=== #


@always_inline
def slice_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: IndexList[MAX_RANK],
    out_shape: IndexList[MAX_RANK],
    starts_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    stops_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    steps_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Call Slice.execute with MAX_RANK tensors.

    Parameters:
        dtype: The data type of the input/output arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        in_shape: Padded input shape (MAX_RANK).
        out_shape: Padded output shape (MAX_RANK).
        starts_ptr: Pointer to padded start indices (int64, length MAX_RANK).
        stops_ptr: Pointer to padded stop indices (int64, length MAX_RANK).
        steps_ptr: Pointer to padded step indices (int64, length MAX_RANK).
        ctx: Device context pointer (null for CPU).
    """
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK, ...].get_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK, ...].get_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )
    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    comptime idx_spec = StaticTensorSpec[DType.int64, 1, ...].get_unknown()

    var starts_tensor = ManagedTensorSlice[io_spec=Input, static_spec=idx_spec](
        starts_ptr, IndexList[1](MAX_RANK)
    )
    var stops_tensor = ManagedTensorSlice[io_spec=Input, static_spec=idx_spec](
        stops_ptr, IndexList[1](MAX_RANK)
    )
    var steps_tensor = ManagedTensorSlice[io_spec=Input, static_spec=idx_spec](
        steps_ptr, IndexList[1](MAX_RANK)
    )

    comptime unknown_starts = create_unknown_int_tuple(MAX_RANK)
    comptime unknown_steps = create_unknown_int_tuple(MAX_RANK)

    if not ctx:
        Slice.execute[
            target="cpu",
            _trace_name="interpreter.slice",
            static_starts=unknown_starts,
            static_steps=unknown_steps,
            dtype=dtype,
            rank=MAX_RANK,
        ](
            output_tensor,
            input_tensor,
            starts_tensor,
            stops_tensor,
            steps_tensor,
            DeviceContextPtr(),
        )
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                Slice.execute[
                    target="gpu",
                    _trace_name="interpreter.slice",
                    static_starts=unknown_starts,
                    static_steps=unknown_steps,
                    dtype=dtype,
                    rank=MAX_RANK,
                ](
                    output_tensor,
                    input_tensor,
                    starts_tensor,
                    stops_tensor,
                    steps_tensor,
                    device_ctx,
                )
            else:
                raise Error(
                    "GPU execution not supported for slice with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


@fieldwise_init
struct _SliceBody(Dispatchable):
    """Dispatch body for the Slice operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var in_shape: IndexList[MAX_RANK]
    var out_shape: IndexList[MAX_RANK]
    var starts_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin]
    var stops_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin]
    var steps_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin]
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        slice_op[t](
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.in_shape,
            self.out_shape,
            self.starts_ptr,
            self.stops_ptr,
            self.steps_ptr,
            self.ctx,
        )


def slice_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    starts_buffer: PythonObject,
    stops_buffer: PythonObject,
    steps_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Slice dispatcher - unwraps PythonObjects and dispatches.

    Pads shapes to MAX_RANK with leading 1s and dispatches a single MAX_RANK
    slice operation. The starts/stops/steps buffers must already be padded
    to MAX_RANK length.

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        starts_buffer: 1D int64 buffer with padded start indices.
        stops_buffer: 1D int64 buffer with padded stop indices.
        steps_buffer: 1D int64 buffer with padded step indices.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var in_shape_obj = in_buffer.shape
    var out_shape_obj = out_buffer.shape
    var in_rank = Int(py=len(in_shape_obj))
    var out_rank = Int(py=len(out_shape_obj))
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    # Validate ranks
    if in_rank > MAX_RANK or out_rank > MAX_RANK:
        raise Error(
            "Unsupported rank for slice: in_rank="
            + String(in_rank)
            + ", out_rank="
            + String(out_rank)
            + ". Max supported rank is "
            + String(MAX_RANK)
        )

    # Pad shapes to MAX_RANK with leading 1s
    var padded_in_shape = _pad_shape_to_max_rank(in_shape_obj, in_rank)
    var padded_out_shape = _pad_shape_to_max_rank(out_shape_obj, out_rank)

    # Get pointers to padded starts/stops/steps (already padded by Python)
    var starts_ptr = _get_buffer_ptr[DType.int64](starts_buffer)
    var stops_ptr = _get_buffer_ptr[DType.int64](stops_buffer)
    var steps_ptr = _get_buffer_ptr[DType.int64](steps_buffer)

    dispatch_dtype(
        _SliceBody(
            out_addr,
            in_addr,
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        ),
        dtype,
    )
