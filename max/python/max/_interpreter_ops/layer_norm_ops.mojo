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

"""Mojo kernel wrappers for layer_norm MO interpreter operations."""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator

from algorithm.functional import IndexList
from math import sqrt
from memory import OpaquePointer
from runtime.asyncrt import DeviceContextPtr
from tensor.managed_tensor_slice import ManagedTensorSlice
from tensor.io_spec import Input
from compiler_internal import StaticTensorSpec
from nn.normalization import layer_norm as nn_layer_norm

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, _get_shape, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_layer_norm_ops() -> PythonObject:
    """Create a Python module with layer_norm kernel function bindings."""
    try:
        var b = PythonModuleBuilder("layer_norm_ops")

        b.def_function[layer_norm_dispatcher](
            "LayerNorm", docstring="Layer normalization"
        )

        return b.finalize()
    except e:
        abort(String("failed to create layer_norm op bindings module: ", e))


# =============================================================================
# Kernel implementations
# =============================================================================


fn _layer_norm_cpu[
    dtype: DType,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch_dim: Int,
    feature_dim: Int,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    """CPU layer normalization on a rank-2 [batch, feature_dim] buffer.

    Uses a 3-pass algorithm:
    1. Compute mean along the feature dimension
    2. Compute variance along the feature dimension
    3. Normalize: (x - mean) / sqrt(var + eps) * gamma + beta

    Parameters:
        dtype: The data type (must be floating point).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        gamma_ptr: Pointer to the gamma buffer (1D, size = feature_dim).
        beta_ptr: Pointer to the beta buffer (1D, size = feature_dim).
        batch_dim: Number of rows (batch dimension).
        feature_dim: Size of the feature dimension (last axis).
        epsilon: Small constant for numerical stability.
    """
    for row in range(batch_dim):
        var offset = row * feature_dim

        # Pass 1: compute mean
        var sum_val = Scalar[dtype](0)
        for i in range(feature_dim):
            sum_val += in_ptr[offset + i]
        var mean_val = sum_val / Scalar[dtype](feature_dim)

        # Pass 2: compute variance
        var var_val = Scalar[dtype](0)
        for i in range(feature_dim):
            var diff = in_ptr[offset + i] - mean_val
            var_val += diff * diff
        var_val = var_val / Scalar[dtype](feature_dim)

        # Pass 3: normalize
        var inv_std = Scalar[dtype](1) / sqrt(var_val + epsilon)
        for i in range(feature_dim):
            out_ptr[offset + i] = (
                in_ptr[offset + i] - mean_val
            ) * inv_std * gamma_ptr[i] + beta_ptr[i]


fn layer_norm_op[
    dtype: DType,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    shape: IndexList[2],
    gamma_shape: IndexList[1],
    epsilon: Scalar[dtype],
    ctx: OpaquePointer[MutExternalOrigin],
) raises where dtype.is_floating_point():
    """Layer normalization on a rank-2 normalized tensor.

    The input is normalized to rank-2 [batch, feature_dim] with layer norm
    applied along the last axis.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        gamma_ptr: Pointer to the gamma buffer (1D).
        beta_ptr: Pointer to the beta buffer (1D).
        shape: The normalized rank-2 shape [batch_dim, feature_dim].
        gamma_shape: The gamma shape [feature_dim].
        epsilon: Small constant for numerical stability.
        ctx: Device context pointer (null for CPU).
    """
    var batch_dim = shape[0]
    var feature_dim = shape[1]

    if not ctx:
        # CPU path: use direct implementation to avoid runtime dependency
        # (nn.normalization requires AsyncRT parallelism which isn't
        # available in the interpreter context)
        _layer_norm_cpu[dtype](
            out_ptr,
            in_ptr,
            gamma_ptr,
            beta_ptr,
            batch_dim,
            feature_dim,
            epsilon,
        )
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype in (DType.float32, DType.float16, DType.bfloat16):
                # GPU path: use nn.normalization.layer_norm kernel via
                # callback functions (similar to softmax_ops.mojo pattern)

                @always_inline
                @parameter
                @__copy_capture(in_ptr, feature_dim)
                fn input_fn[
                    width: Int, rank: Int
                ](coords: IndexList[rank]) -> SIMD[dtype, width]:
                    var c = rebind[IndexList[2]](coords)
                    var flat_idx = c[0] * feature_dim + c[1]
                    return in_ptr.load[width=width](flat_idx)

                @always_inline
                @parameter
                @__copy_capture(gamma_ptr)
                fn gamma_fn[
                    width: Int, rank: Int
                ](coords: IndexList[rank]) -> SIMD[dtype, width]:
                    var c = rebind[IndexList[1]](coords)
                    return gamma_ptr.load[width=width](c[0])

                @always_inline
                @parameter
                @__copy_capture(out_ptr, feature_dim)
                fn output_fn[
                    width: Int, rank: Int, alignment: Int
                ](coords: IndexList[rank], val: SIMD[dtype, width]):
                    var c = rebind[IndexList[2]](coords)
                    var flat_idx = c[0] * feature_dim + c[1]
                    out_ptr.store[width=width](flat_idx, val)

                # Create beta as InputTensor -> TileTensor for the kernel
                comptime beta_spec = StaticTensorSpec[dtype, 1].create_unknown()
                var beta_tensor = ManagedTensorSlice[
                    io_spec=Input, static_spec=beta_spec
                ](beta_ptr, gamma_shape)

                var device_ctx = DeviceContextPtr(ctx)

                nn_layer_norm[
                    dtype,
                    2,
                    input_fn,
                    gamma_fn,
                    output_fn,
                    target="gpu",
                ](
                    shape,
                    gamma_shape,
                    beta_tensor.to_tile_tensor[DType.int64](),
                    epsilon,
                    device_ctx,
                )

                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for layer_norm with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# =============================================================================
# Dispatcher
# =============================================================================


fn layer_norm_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    gamma_buffer: PythonObject,
    beta_buffer: PythonObject,
    epsilon_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Layer normalization dispatcher with dtype dispatch.

    Normalizes the input to rank-2 [batch, feature_dim] and dispatches by dtype.
    The epsilon buffer is a scalar tensor on CPU with the same dtype as input.
    """
    var dtype = _get_dtype(in_buffer)
    var ctx = _get_ctx(device_context_ptr)

    # Extract input shape
    var in_shape_py = in_buffer.shape
    var rank = Int(py=len(in_shape_py))
    var in_shape = _get_shape(in_shape_py, rank)

    # Normalize to rank-2: [batch_dim, feature_dim]
    var feature_dim = in_shape[rank - 1]
    var batch_dim = 1
    for i in range(rank - 1):
        batch_dim *= in_shape[i]

    var normalized_shape = IndexList[2](batch_dim, feature_dim)
    var gamma_shape = IndexList[1](feature_dim)

    # Dispatch by dtype (float only)
    # Extract epsilon scalar directly from the buffer pointer (handles
    # bfloat16 which numpy doesn't support).
    if dtype == DType.float16:
        layer_norm_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            _get_buffer_ptr[DType.float16](gamma_buffer),
            _get_buffer_ptr[DType.float16](beta_buffer),
            normalized_shape,
            gamma_shape,
            _get_buffer_ptr[DType.float16](epsilon_buffer)[0],
            ctx,
        )
    elif dtype == DType.float32:
        layer_norm_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            _get_buffer_ptr[DType.float32](gamma_buffer),
            _get_buffer_ptr[DType.float32](beta_buffer),
            normalized_shape,
            gamma_shape,
            _get_buffer_ptr[DType.float32](epsilon_buffer)[0],
            ctx,
        )
    elif dtype == DType.float64:
        layer_norm_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](in_buffer),
            _get_buffer_ptr[DType.float64](gamma_buffer),
            _get_buffer_ptr[DType.float64](beta_buffer),
            normalized_shape,
            gamma_shape,
            _get_buffer_ptr[DType.float64](epsilon_buffer)[0],
            ctx,
        )
    elif dtype == DType.bfloat16:
        layer_norm_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            _get_buffer_ptr[DType.bfloat16](gamma_buffer),
            _get_buffer_ptr[DType.bfloat16](beta_buffer),
            normalized_shape,
            gamma_shape,
            _get_buffer_ptr[DType.bfloat16](epsilon_buffer)[0],
            ctx,
        )
    else:
        raise Error("Unsupported dtype for layer_norm: " + String(dtype))
