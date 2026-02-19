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

"""Mojo kernel wrappers for transpose MO interpreter operations."""

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
from buffer.dimlist import DimList
from MOGGKernelAPI.MOGGKernelAPI import Transpose

from op_utils import _get_dtype, _get_ctx, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_transpose_ops() -> PythonObject:
    """Create a Python module with transpose kernel function bindings."""
    try:
        var b = PythonModuleBuilder("transpose_ops")

        b.def_function[transpose_dispatcher](
            "Transpose", docstring="Transpose operation"
        )

        return b.finalize()
    except e:
        abort(String("failed to create transpose op bindings module: ", e))


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


# =============================================================================
# Kernel implementation
# =============================================================================


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


# =============================================================================
# Dispatcher
# =============================================================================


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

    @always_inline
    fn _make_ptr[
        dt: DType
    ](addr: Int) -> UnsafePointer[Scalar[dt], MutExternalOrigin]:
        return UnsafePointer[Scalar[dt], MutExternalOrigin](
            unsafe_from_address=addr
        )

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
