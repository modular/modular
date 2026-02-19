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

"""Mojo kernel wrappers for softmax MO interpreter operations."""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator, simd_width_of

from math import exp, log
from algorithm.functional import IndexList
from memory import OpaquePointer
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout
from nn.softmax import softmax as nn_softmax, logsoftmax as nn_logsoftmax
from runtime.asyncrt import DeviceContextPtr

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, _get_shape, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_softmax_ops() -> PythonObject:
    """Create a Python module with softmax kernel function bindings."""
    try:
        var b = PythonModuleBuilder("softmax_ops")

        b.def_function[softmax_dispatcher](
            "Softmax", docstring="Softmax along axis"
        )
        b.def_function[logsoftmax_dispatcher](
            "LogSoftmax", docstring="LogSoftmax along axis"
        )

        return b.finalize()
    except e:
        abort(String("failed to create softmax op bindings module: ", e))


# =============================================================================
# Kernel implementations
# =============================================================================


fn _softmax_cpu[
    dtype: DType,
    is_logsoftmax: Bool,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch_dim: Int,
    axis_dim: Int,
) where dtype.is_floating_point():
    """CPU softmax/logsoftmax on a rank-2 [batch, axis_dim] buffer.

    Uses a numerically stable 3-pass algorithm:
    1. Find max along axis for numerical stability
    2. Compute exp(x - max) and accumulate sum
    3. Normalize (divide by sum, or subtract log(sum) for logsoftmax)

    Parameters:
        dtype: The data type (must be floating point).
        is_logsoftmax: If True, compute log(softmax(x)).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        batch_dim: Number of rows (batch dimension).
        axis_dim: Size of the softmax axis.
    """
    for row in range(batch_dim):
        var offset = row * axis_dim

        # Pass 1: find max for numerical stability
        var max_val = in_ptr[offset]
        for i in range(1, axis_dim):
            var v = in_ptr[offset + i]
            if v > max_val:
                max_val = v

        # Pass 2: compute exp(x - max) and accumulate sum
        var sum_val = Scalar[dtype](0)
        for i in range(axis_dim):
            var exp_val = exp(in_ptr[offset + i] - max_val)
            out_ptr[offset + i] = exp_val
            sum_val += exp_val

        # Pass 3: normalize
        @parameter
        if is_logsoftmax:
            var log_sum = log(sum_val)
            for i in range(axis_dim):
                out_ptr[offset + i] = in_ptr[offset + i] - max_val - log_sum
        else:
            for i in range(axis_dim):
                out_ptr[offset + i] /= sum_val


fn softmax_op[
    dtype: DType,
    is_logsoftmax: Bool,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    shape: IndexList[2],
    ctx: OpaquePointer[MutExternalOrigin],
) raises where dtype.is_floating_point():
    """Softmax/LogSoftmax operation on a rank-2 normalized tensor.

    The input is normalized to rank-2 [batch, axis_dim] with softmax applied
    along axis=1 (the last axis).

    Parameters:
        dtype: The data type of the arrays.
        is_logsoftmax: If True, compute log(softmax(x)) instead of softmax(x).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        shape: The normalized rank-2 shape [batch_dim, axis_dim].
        ctx: Device context pointer.
    """
    var batch_dim = shape[0]
    var axis_dim = shape[1]

    if not ctx:
        # CPU path: use direct implementation to avoid runtime dependency
        # (nn.softmax requires AsyncRT parallelism_level which isn't
        # available in the interpreter context)
        _softmax_cpu[dtype, is_logsoftmax](out_ptr, in_ptr, batch_dim, axis_dim)
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype in (DType.float32, DType.float16, DType.bfloat16):
                # GPU path: use nn.softmax kernel via input_fn + LayoutTensor
                @always_inline
                @parameter
                @__copy_capture(in_ptr, axis_dim)
                fn input_fn[
                    width: Int, rank: Int
                ](coords: IndexList[rank]) -> SIMD[dtype, width]:
                    var c = rebind[IndexList[2]](coords)
                    var flat_idx = c[0] * axis_dim + c[1]
                    return in_ptr.load[width=width](flat_idx)

                comptime out_layout = Layout.row_major(
                    UNKNOWN_VALUE, UNKNOWN_VALUE
                )
                var rt = RuntimeLayout[out_layout].row_major(shape)
                var output_tensor = LayoutTensor[
                    dtype,
                    out_layout,
                    MutExternalOrigin,
                ](out_ptr, rt)

                var device_ctx = DeviceContextPtr(ctx)

                @parameter
                if is_logsoftmax:
                    nn_logsoftmax[
                        dtype,
                        simd_width_of[dtype](),
                        2,
                        input_fn,
                        target="gpu",
                    ](shape, output_tensor, 1, device_ctx)
                else:
                    nn_softmax[
                        dtype,
                        simd_width_of[dtype](),
                        2,
                        input_fn,
                        target="gpu",
                    ](shape, output_tensor, 1, device_ctx)

                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for softmax with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# =============================================================================
# Dispatchers
# =============================================================================


fn _softmax_dispatch[
    is_logsoftmax: Bool,
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Softmax/LogSoftmax dispatcher with dtype dispatch.

    Normalizes the input to rank-2 [batch, axis_dim] and dispatches by dtype.

    Parameters:
        is_logsoftmax: If True, compute log(softmax(x)).
    """
    var dtype = _get_dtype(in_buffer)
    var axis_val = Int(py=axis)
    var ctx = _get_ctx(device_context_ptr)

    # Extract input shape
    var in_shape_py = in_buffer.shape
    var rank = Int(py=len(in_shape_py))
    var in_shape = _get_shape(in_shape_py, rank)

    # Validate axis is the last dimension (kernel limitation)
    if axis_val != rank - 1:
        raise Error(
            "softmax only supports the last axis, got axis="
            + String(axis_val)
            + " for rank="
            + String(rank)
        )

    # Normalize to rank-2: [batch_dim, axis_dim]
    var axis_dim = in_shape[axis_val]
    var batch_dim = 1
    for i in range(rank - 1):
        batch_dim *= in_shape[i]

    var normalized_shape = IndexList[2](batch_dim, axis_dim)

    # Dispatch by dtype (float only)
    if dtype == DType.float16:
        softmax_op[DType.float16, is_logsoftmax](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.float32:
        softmax_op[DType.float32, is_logsoftmax](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.float64:
        softmax_op[DType.float64, is_logsoftmax](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.bfloat16:
        softmax_op[DType.bfloat16, is_logsoftmax](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            normalized_shape,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for softmax: " + String(dtype))


# Concrete dispatcher functions for def_function registration.


fn softmax_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    _softmax_dispatch[is_logsoftmax=False](
        out_buffer, in_buffer, axis, device_context_ptr
    )


fn logsoftmax_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    _softmax_dispatch[is_logsoftmax=True](
        out_buffer, in_buffer, axis, device_context_ptr
    )
