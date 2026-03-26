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
        _gather_dispatch_i32(
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
        _gather_dispatch_i64(
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


def _gather_dispatch_i32(
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
    var idx_ptr = _make_ptr[DType.int32](idx_addr)
    if dtype == DType.float32:
        gather_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            gather_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](in_addr),
                idx_ptr,
                outer_size,
                axis_size,
                inner_size,
                num_indices,
                ctx,
            )
    elif dtype == DType.float16:
        gather_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.bfloat16:
        gather_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int8:
        gather_op(
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int16:
        gather_op(
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int32:
        gather_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int64:
        gather_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint8:
        gather_op(
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint16:
        gather_op(
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint32:
        gather_op(
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint64:
        gather_op(
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.bool:
        gather_op(
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for gather: " + String(dtype))


def _gather_dispatch_i64(
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
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    if dtype == DType.float32:
        gather_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            gather_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](in_addr),
                idx_ptr,
                outer_size,
                axis_size,
                inner_size,
                num_indices,
                ctx,
            )
    elif dtype == DType.float16:
        gather_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.bfloat16:
        gather_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int8:
        gather_op(
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int16:
        gather_op(
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int32:
        gather_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.int64:
        gather_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint8:
        gather_op(
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint16:
        gather_op(
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint32:
        gather_op(
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.uint64:
        gather_op(
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    elif dtype == DType.bool:
        gather_op(
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_indices,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for gather: " + String(dtype))


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
        _gather_nd_dispatch_i32(
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
        _gather_nd_dispatch_i64(
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


def _gather_nd_dispatch_i32(
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
    var idx_ptr = _make_ptr[DType.int32](idx_addr)
    if dtype == DType.float32:
        gather_nd_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            gather_nd_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](in_addr),
                idx_ptr,
                batch_size,
                indices_outer_size,
                index_depth,
                suffix_size,
                input_data_stride,
                indexed_strides,
                ctx,
            )
    elif dtype == DType.float16:
        gather_nd_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.bfloat16:
        gather_nd_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int8:
        gather_nd_op(
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int16:
        gather_nd_op(
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int32:
        gather_nd_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int64:
        gather_nd_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint8:
        gather_nd_op(
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint16:
        gather_nd_op(
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint32:
        gather_nd_op(
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint64:
        gather_nd_op(
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.bool:
        gather_nd_op(
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for gather_nd: " + String(dtype))


def _gather_nd_dispatch_i64(
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
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    if dtype == DType.float32:
        gather_nd_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            gather_nd_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](in_addr),
                idx_ptr,
                batch_size,
                indices_outer_size,
                index_depth,
                suffix_size,
                input_data_stride,
                indexed_strides,
                ctx,
            )
    elif dtype == DType.float16:
        gather_nd_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.bfloat16:
        gather_nd_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int8:
        gather_nd_op(
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int16:
        gather_nd_op(
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int32:
        gather_nd_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.int64:
        gather_nd_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint8:
        gather_nd_op(
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint16:
        gather_nd_op(
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint32:
        gather_nd_op(
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.uint64:
        gather_nd_op(
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    elif dtype == DType.bool:
        gather_nd_op(
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            idx_ptr,
            batch_size,
            indices_outer_size,
            index_depth,
            suffix_size,
            input_data_stride,
            indexed_strides,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for gather_nd: " + String(dtype))


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
        _scatter_dispatch_i32(
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
        _scatter_dispatch_i64(
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


def _scatter_dispatch_i32(
    dtype: DType,
    out_addr: Int,
    upd_addr: Int,
    idx_addr: Int,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
    num_updates_axis: Int,
) raises:
    var idx_ptr = _make_ptr[DType.int32](idx_addr)
    if dtype == DType.float32:
        scatter_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            scatter_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](upd_addr),
                idx_ptr,
                outer_size,
                axis_size,
                inner_size,
                num_updates_axis,
            )
    elif dtype == DType.float16:
        scatter_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.bfloat16:
        scatter_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int8:
        scatter_op(
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int16:
        scatter_op(
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int32:
        scatter_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int64:
        scatter_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint8:
        scatter_op(
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint16:
        scatter_op(
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint32:
        scatter_op(
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint64:
        scatter_op(
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.bool:
        scatter_op(
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    else:
        raise Error("Unsupported dtype for scatter: " + String(dtype))


def _scatter_dispatch_i64(
    dtype: DType,
    out_addr: Int,
    upd_addr: Int,
    idx_addr: Int,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
    num_updates_axis: Int,
) raises:
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    if dtype == DType.float32:
        scatter_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            scatter_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](upd_addr),
                idx_ptr,
                outer_size,
                axis_size,
                inner_size,
                num_updates_axis,
            )
    elif dtype == DType.float16:
        scatter_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.bfloat16:
        scatter_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int8:
        scatter_op(
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int16:
        scatter_op(
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int32:
        scatter_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.int64:
        scatter_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint8:
        scatter_op(
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint16:
        scatter_op(
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint32:
        scatter_op(
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.uint64:
        scatter_op(
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    elif dtype == DType.bool:
        scatter_op(
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](upd_addr),
            idx_ptr,
            outer_size,
            axis_size,
            inner_size,
            num_updates_axis,
        )
    else:
        raise Error("Unsupported dtype for scatter: " + String(dtype))
