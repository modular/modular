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

"""Mojo kernel wrappers for rms_norm MO interpreter operations."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm.functional import IndexList
from std.math import sqrt
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice
from tensor.io_spec import Input
from compiler_internal import StaticTensorSpec
from nn.normalization import rms_norm as nn_rms_norm

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, _get_shape, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_rms_norm_ops() -> PythonObject:
    """Create a Python module with rms_norm kernel function bindings."""
    try:
        var b = PythonModuleBuilder("rms_norm_ops")

        b.def_function[rms_norm_dispatcher](
            "RmsNorm", docstring="RMS normalization"
        )

        return b.finalize()
    except e:
        abort(t"failed to create rms_norm op bindings module: {e}")


# =============================================================================
# Kernel implementations
# =============================================================================


def _rms_norm_cpu[
    dtype: DType,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch_dim: Int,
    feature_dim: Int,
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    multiply_before_cast: Bool,
) where dtype.is_floating_point():
    """CPU RMS normalization on a rank-2 [batch, feature_dim] buffer.

    Uses a 2-pass algorithm:
    1. Compute root mean square along the feature dimension
    2. Normalize: x / rms * (gamma + weight_offset)

    Parameters:
        dtype: The data type (must be floating point).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        gamma_ptr: Pointer to the weight buffer (1D, size = feature_dim).
        batch_dim: Number of rows (batch dimension).
        feature_dim: Size of the feature dimension (last axis).
        epsilon: Small constant for numerical stability.
        weight_offset: Value added to weight before multiplication.
        multiply_before_cast: Gemma-style (True) vs Llama-style (False).
            On CPU with uniform dtype this has no effect, but kept for
            API parity.
    """
    for row in range(batch_dim):
        var offset = row * feature_dim

        # Pass 1: compute sum of squares
        var sum_sq = Scalar[dtype](0)
        for i in range(feature_dim):
            var val = in_ptr[offset + i]
            sum_sq += val * val
        var rms = sqrt(sum_sq / Scalar[dtype](feature_dim) + epsilon)

        # Pass 2: normalize
        var inv_rms = Scalar[dtype](1) / rms
        for i in range(feature_dim):
            out_ptr[offset + i] = (
                in_ptr[offset + i] * inv_rms * (gamma_ptr[i] + weight_offset)
            )


def rms_norm_op[
    dtype: DType,
    multiply_before_cast: Bool,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    shape: IndexList[2],
    gamma_shape: IndexList[1],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises where dtype.is_floating_point():
    """RMS normalization on a rank-2 normalized tensor.

    Parameters:
        dtype: The data type of the arrays.
        multiply_before_cast: Gemma-style (True) vs Llama-style (False).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        gamma_ptr: Pointer to the weight buffer (1D).
        shape: The normalized rank-2 shape [batch_dim, feature_dim].
        gamma_shape: The gamma shape [feature_dim].
        epsilon: Small constant for numerical stability.
        weight_offset: Value added to weight before multiplication.
        ctx: Device context pointer (null for CPU).
    """
    var batch_dim = shape[0]
    var feature_dim = shape[1]

    if not ctx:
        _rms_norm_cpu[dtype](
            out_ptr,
            in_ptr,
            gamma_ptr,
            batch_dim,
            feature_dim,
            epsilon,
            weight_offset,
            multiply_before_cast,
        )
    else:
        comptime if has_accelerator():
            comptime if dtype in (DType.float32, DType.float16, DType.bfloat16):

                @always_inline
                @parameter
                @__copy_capture(in_ptr, feature_dim)
                def input_fn[
                    width: Int, rank: Int
                ](coords: IndexList[rank]) -> SIMD[dtype, width]:
                    var c = rebind[IndexList[2]](coords)
                    var flat_idx = c[0] * feature_dim + c[1]
                    return in_ptr.load[width=width](flat_idx)

                @always_inline
                @parameter
                @__copy_capture(out_ptr, feature_dim)
                def output_fn[
                    width: Int, rank: Int, alignment: Int
                ](coords: IndexList[rank], val: SIMD[dtype, width]):
                    var c = rebind[IndexList[2]](coords)
                    var flat_idx = c[0] * feature_dim + c[1]
                    out_ptr.store[width=width](flat_idx, val)

                comptime gamma_spec = StaticTensorSpec[
                    dtype, 1, ...
                ].get_unknown()
                var gamma_tensor = ManagedTensorSlice[
                    io_spec=Input, static_spec=gamma_spec
                ](gamma_ptr, gamma_shape)

                var device_ctx = DeviceContextPtr(ctx.unsafe_value())

                nn_rms_norm[
                    dtype,
                    2,
                    input_fn,
                    output_fn,
                    target="gpu",
                    multiply_before_cast=multiply_before_cast,
                ](
                    shape,
                    gamma_tensor.to_tile_tensor[DType.int64](),
                    epsilon,
                    weight_offset,
                    device_ctx,
                )

            else:
                raise Error(
                    "GPU execution not supported for rms_norm with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# =============================================================================
# Dispatcher
# =============================================================================


def _dispatch_rms_norm[
    dtype: DType,
    multiply_before_cast: Bool,
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    gamma_buffer: PythonObject,
    epsilon_buffer: PythonObject,
    weight_offset_buffer: PythonObject,
    in_shape_py: PythonObject,
    rank: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises where dtype.is_floating_point():
    """Type-specialized RMS norm dispatch helper.

    Parameters:
        dtype: Element type (must be floating point).
        multiply_before_cast: Gemma-style (True) vs Llama-style (False).

    Args:
        out_buffer: Output buffer Python object.
        in_buffer: Input buffer Python object.
        gamma_buffer: Weight buffer Python object.
        epsilon_buffer: Epsilon scalar buffer (CPU).
        weight_offset_buffer: Weight offset scalar buffer (CPU).
        in_shape_py: Python tuple of input shape.
        rank: Input rank.
        ctx: Device context pointer.
    """
    var in_shape = _get_shape(in_shape_py, rank)

    var feature_dim = in_shape[rank - 1]
    var batch_dim = 1
    for i in range(rank - 1):
        batch_dim *= in_shape[i]

    var normalized_shape = IndexList[2](batch_dim, feature_dim)
    var gamma_shape = IndexList[1](feature_dim)

    rms_norm_op[dtype, multiply_before_cast](
        _get_buffer_ptr[dtype](out_buffer),
        _get_buffer_ptr[dtype](in_buffer),
        _get_buffer_ptr[dtype](gamma_buffer),
        normalized_shape,
        gamma_shape,
        _get_buffer_ptr[dtype](epsilon_buffer)[0],
        _get_buffer_ptr[dtype](weight_offset_buffer)[0],
        ctx,
    )


def rms_norm_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    gamma_buffer: PythonObject,
    epsilon_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """RMS normalization dispatcher with dtype and multiply_before_cast dispatch.

    Normalizes the input to rank-2 [batch, feature_dim] and dispatches by dtype.
    Epsilon and weight_offset are scalar tensors on CPU with the same dtype as
    input.

    Args:
        out_buffer: Output buffer.
        in_buffer: Input buffer.
        gamma_buffer: Weight buffer.
        epsilon_buffer: Epsilon scalar buffer (CPU).
        params: Python tuple (weight_offset_buffer, multiply_before_cast_flag).
        device_context_ptr: Device context pointer.
    """
    var weight_offset_buffer = params[0]
    var dtype = _get_dtype(in_buffer)
    var ctx = _get_ctx(device_context_ptr)
    var mbc = Bool(Int(py=params[1]))

    var in_shape_py = in_buffer.shape
    var rank = Int(py=len(in_shape_py))

    if mbc:
        if dtype == DType.float16:
            _dispatch_rms_norm[DType.float16, True](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        elif dtype == DType.float32:
            _dispatch_rms_norm[DType.float32, True](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        elif dtype == DType.float64:
            _dispatch_rms_norm[DType.float64, True](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        elif dtype == DType.bfloat16:
            _dispatch_rms_norm[DType.bfloat16, True](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        else:
            raise Error("Unsupported dtype for rms_norm: " + String(dtype))
    else:
        if dtype == DType.float16:
            _dispatch_rms_norm[DType.float16, False](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        elif dtype == DType.float32:
            _dispatch_rms_norm[DType.float32, False](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        elif dtype == DType.float64:
            _dispatch_rms_norm[DType.float64, False](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        elif dtype == DType.bfloat16:
            _dispatch_rms_norm[DType.bfloat16, False](
                out_buffer,
                in_buffer,
                gamma_buffer,
                epsilon_buffer,
                weight_offset_buffer,
                in_shape_py,
                rank,
                ctx,
            )
        else:
            raise Error("Unsupported dtype for rms_norm: " + String(dtype))
