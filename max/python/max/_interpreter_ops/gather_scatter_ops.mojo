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

"""Mojo kernel wrappers for gather/scatter MO interpreter operations."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm.functional import elementwise, IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from std.sys.info import has_apple_gpu_accelerator

from op_utils import (
    _get_dtype,
    _get_ctx,
    _get_shape,
    _make_ptr,
    dispatch_dtype,
    Dispatchable,
    MAX_RANK,
)


@export
def PyInit_gather_scatter_ops() -> PythonObject:
    """Create a Python module with gather/scatter kernel function bindings."""
    try:
        var b = PythonModuleBuilder("gather_scatter_ops")
        b.def_function[gather_dispatcher](
            "Gather", docstring="Gather along axis"
        )
        b.def_function[gather_nd_dispatcher](
            "GatherNd", docstring="Gather with N-dimensional indices"
        )
        b.def_function[scatter_dispatcher](
            "Scatter", docstring="Scatter along axis"
        )
        return b.finalize()
    except e:
        abort(t"failed to create gather scatter op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Gather operation
# ===----------------------------------------------------------------------=== #
#
# Normalizes input to 3D: [outer_size, axis_size, inner_size]
# Output is [outer_size, num_indices, inner_size]
# For each output element: output[o, i, k] = input[o, indices[i], k]


@always_inline
def gather_op[
    dtype: DType, idx_dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    indices_ptr: UnsafePointer[Scalar[idx_dtype], MutExternalOrigin],
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
    num_indices: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    var total = outer_size * num_indices * inner_size
    var in_axis_stride = axis_size * inner_size
    var out_axis_stride = num_indices * inner_size

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        indices_ptr,
        in_axis_stride,
        out_axis_stride,
        inner_size,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var outer_idx, rem = divmod(i, out_axis_stride)
        var idx_pos, inner_idx = divmod(rem, inner_size)
        var gather_idx = Int(indices_ptr[idx_pos])
        var in_flat = (
            outer_idx * in_axis_stride + gather_idx * inner_size + inner_idx
        )
        out_ptr[i] = in_ptr[in_flat]

    if not ctx:
        elementwise[func, simd_width=1](IndexList[1](total))
    else:
        comptime if has_accelerator():
            var device_ctx = DeviceContextPtr(ctx)
            elementwise[func, simd_width=1, target="gpu"](
                IndexList[1](total), device_ctx
            )
        else:
            raise Error("No GPU accelerator available")


def gather_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    indices_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Gather dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer.
        in_buffer: Input data buffer.
        indices_buffer: Indices buffer (int32 or int64).
        params: Python tuple (axis, outer_size, axis_size, inner_size,
            num_indices).
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var idx_dtype = _get_dtype(indices_buffer)
    var outer_size = Int(py=params[0])
    var axis_size = Int(py=params[1])
    var inner_size = Int(py=params[2])
    var num_indices = Int(py=params[3])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var idx_addr = Int(py=indices_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    if idx_dtype == DType.int32:
        _gather_dispatch_integer[DType.int32](
            dtype,
            out_addr,
            in_addr,
            idx_addr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif idx_dtype == DType.int64:
        _gather_dispatch_integer[DType.int64](
            dtype,
            out_addr,
            in_addr,
            idx_addr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    else:
        raise Error("Unsupported index dtype for gather: " + String(idx_dtype))


struct _GatherBody[idx_dtype: DType](Dispatchable):
    """Dispatch body for the Gather operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var idx_ptr: UnsafePointer[Scalar[Self.idx_dtype], MutExternalOrigin]
    var outer_size: Int
    var axis_size: Int
    var inner_size: Int
    var num_indices: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def __init__(
        out self,
        out_addr: Int,
        in_addr: Int,
        idx_ptr: UnsafePointer[Scalar[Self.idx_dtype], MutExternalOrigin],
        outer_size: Int,
        axis_size: Int,
        inner_size: Int,
        num_indices: Int,
        ctx: OpaquePointer[MutExternalOrigin],
    ):
        self.out_addr = out_addr
        self.in_addr = in_addr
        self.idx_ptr = idx_ptr
        self.outer_size = outer_size
        self.axis_size = axis_size
        self.inner_size = inner_size
        self.num_indices = num_indices
        self.ctx = ctx

    def call[t: DType](self) raises -> None:
        gather_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.idx_ptr,
            self.outer_size,
            self.axis_size,
            self.inner_size,
            self.num_indices,
            self.ctx,
        )


def _gather_dispatch_integer[
    d: DType
](
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    idx_addr: Int,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
    num_indices: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    dispatch_dtype(
        _GatherBody[d](
            out_addr,
            in_addr,
            _make_ptr[d](idx_addr),
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# GatherNd operation
# ===----------------------------------------------------------------------=== #
#
# Flattened to: [batch_size, indices_outer_size, suffix_size]
# indices are [batch_size, indices_outer_size, index_depth]
# For each output element:
#   1. Determine batch, index-outer, and suffix positions
#   2. Read the index vector from indices
#   3. Compute flat input position using pre-computed strides


@always_inline
def gather_nd_op[
    dtype: DType, idx_dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    indices_ptr: UnsafePointer[Scalar[idx_dtype], MutExternalOrigin],
    batch_size: Int,
    indices_outer_size: Int,
    index_depth: Int,
    suffix_size: Int,
    input_data_stride: Int,
    indexed_strides: InlineArray[Int, MAX_RANK],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    var total = batch_size * indices_outer_size * suffix_size
    var out_batch_stride = indices_outer_size * suffix_size
    var idx_batch_stride = indices_outer_size * index_depth

    # Unpack strides to individual captures for GPU compatibility.
    var s0 = indexed_strides[0]
    var s1 = indexed_strides[1]
    var s2 = indexed_strides[2]
    var s3 = indexed_strides[3]
    var s4 = indexed_strides[4]

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        indices_ptr,
        out_batch_stride,
        idx_batch_stride,
        index_depth,
        suffix_size,
        input_data_stride,
        s0,
        s1,
        s2,
        s3,
        s4,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var batch_idx, rem = divmod(i, out_batch_stride)
        var indices_outer_idx, suffix_idx = divmod(rem, suffix_size)

        var in_offset = batch_idx * input_data_stride
        var idx_base = batch_idx * idx_batch_stride + (
            indices_outer_idx * index_depth
        )

        # Unrolled loop over index_depth (max MAX_RANK=5) to avoid
        # dynamic loop and InlineArray capture in GPU kernels.
        if index_depth >= 1:
            in_offset += Int(indices_ptr[idx_base]) * s0
        if index_depth >= 2:
            in_offset += Int(indices_ptr[idx_base + 1]) * s1
        if index_depth >= 3:
            in_offset += Int(indices_ptr[idx_base + 2]) * s2
        if index_depth >= 4:
            in_offset += Int(indices_ptr[idx_base + 3]) * s3
        if index_depth >= 5:
            in_offset += Int(indices_ptr[idx_base + 4]) * s4

        in_offset += suffix_idx
        out_ptr[i] = in_ptr[in_offset]

    if not ctx:
        elementwise[func, simd_width=1](IndexList[1](total))
    else:
        comptime if has_accelerator():
            var device_ctx = DeviceContextPtr(ctx)
            elementwise[func, simd_width=1, target="gpu"](
                IndexList[1](total), device_ctx
            )
        else:
            raise Error("No GPU accelerator available")


def gather_nd_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    indices_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """GatherNd dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer.
        in_buffer: Input data buffer.
        indices_buffer: Indices buffer (int32 or int64).
        params: Python tuple (batch_size, indices_outer_size, index_depth,
            suffix_size, input_data_stride, input_inner_shape).
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var idx_dtype = _get_dtype(indices_buffer)
    var b_size = Int(py=params[0])
    var io_size = Int(py=params[1])
    var i_depth = Int(py=params[2])
    var s_size = Int(py=params[3])
    var id_stride = Int(py=params[4])
    var input_inner_shape = params[5]
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var idx_addr = Int(py=indices_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    var inner_rank = Int(py=len(input_inner_shape))
    var indexed_strides = InlineArray[Int, MAX_RANK](fill=0)
    var stride = 1
    for j in range(inner_rank - 1, -1, -1):
        indexed_strides[j] = stride
        stride *= Int(py=input_inner_shape[j])

    if idx_dtype == DType.int32:
        _gather_nd_dispatch_integer[DType.int32](
            dtype,
            out_addr,
            in_addr,
            idx_addr,
            b_size,
            io_size,
            i_depth,
            s_size,
            id_stride,
            indexed_strides,
            ctx,
        )
    elif idx_dtype == DType.int64:
        _gather_nd_dispatch_integer[DType.int64](
            dtype,
            out_addr,
            in_addr,
            idx_addr,
            b_size,
            io_size,
            i_depth,
            s_size,
            id_stride,
            indexed_strides,
            ctx,
        )
    else:
        raise Error(
            "Unsupported index dtype for gather_nd: " + String(idx_dtype)
        )


struct _GatherNdBody[idx_dtype: DType](Dispatchable):
    """Dispatch body for the GatherNd operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var idx_ptr: UnsafePointer[Scalar[Self.idx_dtype], MutExternalOrigin]
    var batch_size: Int
    var indices_outer_size: Int
    var index_depth: Int
    var suffix_size: Int
    var input_data_stride: Int
    var indexed_strides: InlineArray[Int, MAX_RANK]
    var ctx: OpaquePointer[MutExternalOrigin]

    def __init__(
        out self,
        out_addr: Int,
        in_addr: Int,
        idx_ptr: UnsafePointer[Scalar[Self.idx_dtype], MutExternalOrigin],
        batch_size: Int,
        indices_outer_size: Int,
        index_depth: Int,
        suffix_size: Int,
        input_data_stride: Int,
        indexed_strides: InlineArray[Int, MAX_RANK],
        ctx: OpaquePointer[MutExternalOrigin],
    ):
        self.out_addr = out_addr
        self.in_addr = in_addr
        self.idx_ptr = idx_ptr
        self.batch_size = batch_size
        self.indices_outer_size = indices_outer_size
        self.index_depth = index_depth
        self.suffix_size = suffix_size
        self.input_data_stride = input_data_stride
        self.indexed_strides = indexed_strides
        self.ctx = ctx

    def call[t: DType](self) raises -> None:
        gather_nd_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.idx_ptr,
            self.batch_size,
            self.indices_outer_size,
            self.index_depth,
            self.suffix_size,
            self.input_data_stride,
            self.indexed_strides,
            self.ctx,
        )


def _gather_nd_dispatch_integer[
    d: DType
](
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    idx_addr: Int,
    batch_size: Int,
    indices_outer_size: Int,
    index_depth: Int,
    suffix_size: Int,
    input_data_stride: Int,
    indexed_strides: InlineArray[Int, MAX_RANK],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    dispatch_dtype(
        _GatherNdBody[d](
            out_addr,
            in_addr,
            _make_ptr[d](idx_addr),
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# Scatter operation
# ===----------------------------------------------------------------------=== #
#
# Normalizes input to 3D: [outer_size, axis_size, inner_size]
# Updates/indices have shape: [outer_size, num_updates_axis, inner_size]
# Output is initialized as a copy of input (done by the Python handler).
# For each update position (o, u, k):
#   output[o, indices[o, u, k], k] = updates[o, u, k]


@always_inline
def scatter_op[
    dtype: DType, idx_dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    updates_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    indices_ptr: UnsafePointer[Scalar[idx_dtype], MutExternalOrigin],
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
    num_updates_axis: Int,
) raises:
    """Write scattered updates into the output buffer along the axis.

    The output must already contain a copy of the input tensor.
    This kernel only writes positions specified by indices.
    CPU-only (mo.scatter is MO_HostOnly).

    Parameters:
        dtype: Data type of the data tensors (inferred from pointers).
        idx_dtype: Data type of the index tensor (inferred from pointer).

    Args:
        out_ptr: Output buffer (pre-filled with input copy).
        updates_ptr: Values to scatter, flat length outer*num_updates_axis*inner.
        indices_ptr: Axis indices, same shape as updates.
        outer_size: Product of dims before the scatter axis.
        axis_size: Size of the scatter axis in the output.
        inner_size: Product of dims after the scatter axis.
        num_updates_axis: Size of the scatter axis in updates/indices.
    """
    var total = outer_size * num_updates_axis * inner_size
    var out_axis_stride = axis_size * inner_size
    var upd_axis_stride = num_updates_axis * inner_size

    for i in range(total):
        var outer_idx, rem = divmod(i, upd_axis_stride)
        var _, inner_idx = divmod(rem, inner_size)
        var scatter_idx = Int(indices_ptr[i])
        var out_flat = (
            outer_idx * out_axis_stride + scatter_idx * inner_size + inner_idx
        )
        out_ptr[out_flat] = updates_ptr[i]


def scatter_dispatcher(
    out_buffer: PythonObject,
    updates_buffer: PythonObject,
    indices_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Scatter dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer (pre-filled with input copy).
        updates_buffer: Updates data buffer.
        indices_buffer: Indices buffer (int32 or int64).
        params: Python tuple (outer_size, axis_size, inner_size,
            num_updates_axis).
        device_context_ptr: Device context pointer (unused, CPU-only).
    """
    var dtype = _get_dtype(updates_buffer)
    var idx_dtype = _get_dtype(indices_buffer)
    var outer_size = Int(py=params[0])
    var axis_size = Int(py=params[1])
    var inner_size = Int(py=params[2])
    var num_updates_axis = Int(py=params[3])
    var out_addr = Int(py=out_buffer._data_ptr())
    var upd_addr = Int(py=updates_buffer._data_ptr())
    var idx_addr = Int(py=indices_buffer._data_ptr())

    if idx_dtype == DType.int32:
        _scatter_dispatch_integer[DType.int32](
            dtype,
            out_addr,
            upd_addr,
            idx_addr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif idx_dtype == DType.int64:
        _scatter_dispatch_integer[DType.int64](
            dtype,
            out_addr,
            upd_addr,
            idx_addr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    else:
        raise Error("Unsupported index dtype for scatter: " + String(idx_dtype))


struct _ScatterBody[idx_dtype: DType](Dispatchable):
    """Dispatch body for the Scatter operation over data dtypes."""

    var out_addr: Int
    var upd_addr: Int
    var idx_ptr: UnsafePointer[Scalar[Self.idx_dtype], MutExternalOrigin]
    var outer_size: Int
    var axis_size: Int
    var inner_size: Int
    var num_updates_axis: Int

    def __init__(
        out self,
        out_addr: Int,
        upd_addr: Int,
        idx_ptr: UnsafePointer[Scalar[Self.idx_dtype], MutExternalOrigin],
        outer_size: Int,
        axis_size: Int,
        inner_size: Int,
        num_updates_axis: Int,
    ):
        self.out_addr = out_addr
        self.upd_addr = upd_addr
        self.idx_ptr = idx_ptr
        self.outer_size = outer_size
        self.axis_size = axis_size
        self.inner_size = inner_size
        self.num_updates_axis = num_updates_axis

    def call[t: DType](self) raises -> None:
        scatter_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.upd_addr),
            self.idx_ptr,
            self.outer_size,
            self.axis_size,
            self.inner_size,
            self.num_updates_axis,
        )


def _scatter_dispatch_integer[
    d: DType
](
    dtype: DType,
    out_addr: Int,
    upd_addr: Int,
    idx_addr: Int,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
    num_updates_axis: Int,
) raises:
    dispatch_dtype(
        _ScatterBody[d](
            out_addr,
            upd_addr,
            _make_ptr[d](idx_addr),
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        ),
        dtype,
    )
