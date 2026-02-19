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

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator, simd_width_of

from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from runtime.asyncrt import DeviceContextPtr
from tensor.managed_tensor_slice import ManagedTensorSlice
from tensor.io_spec import Input, Output
from compiler_internal import StaticTensorSpec
from buffer.dimlist import DimList
from MOGGKernelAPI.MOGGKernelAPI import Slice, StaticBroadcastTo, Transpose

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, _get_size, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_data_movement_ops() -> PythonObject:
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
        abort(String("failed to create data movement op bindings module: ", e))


# =============================================================================
# Helpers
# =============================================================================


fn _pad_shape_to_max_rank(
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


@always_inline
fn _make_ptr[
    dtype: DType
](addr: Int) -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
    return UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=addr
    )


# ===----------------------------------------------------------------------=== #
# Broadcast operation
# ===----------------------------------------------------------------------=== #


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
    var padded_in_shape = _pad_shape_to_max_rank(in_shape_obj, in_rank)
    var padded_out_shape = _pad_shape_to_max_rank(out_shape_obj, out_rank)

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


# ===----------------------------------------------------------------------=== #
# Transpose operation
# ===----------------------------------------------------------------------=== #


@always_inline
fn transpose_op[
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
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()
    comptime perm_spec = StaticTensorSpec[DType.int64, 1].create_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )

    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    # Create permutation tensor from stack data
    var perm_ptr = UnsafePointer[Scalar[DType.int64], MutExternalOrigin](
        unsafe_from_address=Int(UnsafePointer(to=perm_data[0]))
    )
    var perm_tensor = ManagedTensorSlice[io_spec=Input, static_spec=perm_spec](
        perm_ptr, IndexList[1](MAX_RANK)
    )

    if not ctx:
        Transpose.execute[
            target="cpu",
            _trace_name="interpreter.transpose",
            static_permutations = DimList.create_unknown[MAX_RANK](),
            dtype=dtype,
            rank=MAX_RANK,
        ](output_tensor, input_tensor, perm_tensor, DeviceContextPtr())
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                Transpose.execute[
                    target="gpu",
                    _trace_name="interpreter.transpose",
                    static_permutations = DimList.create_unknown[MAX_RANK](),
                    dtype=dtype,
                    rank=MAX_RANK,
                ](output_tensor, input_tensor, perm_tensor, device_ctx)
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for transpose"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


fn transpose_dispatcher(
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

    # Dispatch by dtype
    if dtype == DType.float32:
        transpose_op[DType.float32](
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.float64:
        transpose_op[DType.float64](
            _make_ptr[DType.float64](out_addr),
            _make_ptr[DType.float64](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.float16:
        transpose_op[DType.float16](
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.bfloat16:
        transpose_op[DType.bfloat16](
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.int8:
        transpose_op[DType.int8](
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.int16:
        transpose_op[DType.int16](
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.int32:
        transpose_op[DType.int32](
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.int64:
        transpose_op[DType.int64](
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.uint8:
        transpose_op[DType.uint8](
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.uint16:
        transpose_op[DType.uint16](
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.uint32:
        transpose_op[DType.uint32](
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.uint64:
        transpose_op[DType.uint64](
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    elif dtype == DType.bool:
        transpose_op[DType.bool](
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            padded_in_shape,
            padded_out_shape,
            padded_perm,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for transpose: " + String(dtype))


# ===----------------------------------------------------------------------=== #
# Memcpy operation (copy elements between buffers with offsets)
# ===----------------------------------------------------------------------=== #


fn memcpy_dispatcher(
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

    if dtype == DType.float16:
        memcpy_op[DType.float16](
            _get_buffer_ptr[DType.float16](dst_buffer),
            _get_buffer_ptr[DType.float16](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.float32:
        memcpy_op[DType.float32](
            _get_buffer_ptr[DType.float32](dst_buffer),
            _get_buffer_ptr[DType.float32](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.float64:
        memcpy_op[DType.float64](
            _get_buffer_ptr[DType.float64](dst_buffer),
            _get_buffer_ptr[DType.float64](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.bfloat16:
        memcpy_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](dst_buffer),
            _get_buffer_ptr[DType.bfloat16](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.int8:
        memcpy_op[DType.int8](
            _get_buffer_ptr[DType.int8](dst_buffer),
            _get_buffer_ptr[DType.int8](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.int16:
        memcpy_op[DType.int16](
            _get_buffer_ptr[DType.int16](dst_buffer),
            _get_buffer_ptr[DType.int16](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.int32:
        memcpy_op[DType.int32](
            _get_buffer_ptr[DType.int32](dst_buffer),
            _get_buffer_ptr[DType.int32](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.int64:
        memcpy_op[DType.int64](
            _get_buffer_ptr[DType.int64](dst_buffer),
            _get_buffer_ptr[DType.int64](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.uint8:
        memcpy_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](dst_buffer),
            _get_buffer_ptr[DType.uint8](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.uint16:
        memcpy_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](dst_buffer),
            _get_buffer_ptr[DType.uint16](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.uint32:
        memcpy_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](dst_buffer),
            _get_buffer_ptr[DType.uint32](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.uint64:
        memcpy_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](dst_buffer),
            _get_buffer_ptr[DType.uint64](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    elif dtype == DType.bool:
        memcpy_op[DType.bool](
            _get_buffer_ptr[DType.bool](dst_buffer),
            _get_buffer_ptr[DType.bool](src_buffer),
            d_off,
            s_off,
            cnt,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for memcpy: " + String(dtype))


@always_inline
fn memcpy_op[
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
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        d.store[width=width](i, s.load[width=width](i))

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](count))
    else:
        # GPU execution
        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](count), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
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
fn slice_op[
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
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )
    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    comptime idx_spec = StaticTensorSpec[DType.int64, 1].create_unknown()

    var starts_tensor = ManagedTensorSlice[io_spec=Input, static_spec=idx_spec](
        starts_ptr, IndexList[1](MAX_RANK)
    )
    var stops_tensor = ManagedTensorSlice[io_spec=Input, static_spec=idx_spec](
        stops_ptr, IndexList[1](MAX_RANK)
    )
    var steps_tensor = ManagedTensorSlice[io_spec=Input, static_spec=idx_spec](
        steps_ptr, IndexList[1](MAX_RANK)
    )

    comptime unknown_steps = DimList.create_unknown[MAX_RANK]()

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl
        Slice.execute[
            target="cpu",
            _trace_name="interpreter.slice",
            static_steps=unknown_steps,
            dtype=dtype,
            rank=MAX_RANK,
            use_blocking_impl=True,
        ](
            output_tensor,
            input_tensor,
            starts_tensor,
            stops_tensor,
            steps_tensor,
            DeviceContextPtr(),
        )
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                Slice.execute[
                    target="gpu",
                    _trace_name="interpreter.slice",
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
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for slice with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


fn slice_dispatcher(
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

    # Dispatch by dtype
    if dtype == DType.float32:
        slice_op[DType.float32](
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.float64:
        slice_op[DType.float64](
            _make_ptr[DType.float64](out_addr),
            _make_ptr[DType.float64](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.float16:
        slice_op[DType.float16](
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.bfloat16:
        slice_op[DType.bfloat16](
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.int8:
        slice_op[DType.int8](
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.int16:
        slice_op[DType.int16](
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.int32:
        slice_op[DType.int32](
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.int64:
        slice_op[DType.int64](
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.uint8:
        slice_op[DType.uint8](
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.uint16:
        slice_op[DType.uint16](
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.uint32:
        slice_op[DType.uint32](
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.uint64:
        slice_op[DType.uint64](
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    elif dtype == DType.bool:
        slice_op[DType.bool](
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            padded_in_shape,
            padded_out_shape,
            starts_ptr,
            stops_ptr,
            steps_ptr,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for slice: " + String(dtype))
