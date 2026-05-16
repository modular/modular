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

"""Mojo kernel wrappers for group_norm MO interpreter operations."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm.functional import IndexList
from std.math import sqrt
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice
from tensor.io_spec import Input, Output
from compiler_internal import StaticTensorSpec
from nn.normalization import group_norm as nn_group_norm

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, _get_shape, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_group_norm_ops() -> PythonObject:
    """Create a Python module with group_norm kernel function bindings."""
    try:
        var b = PythonModuleBuilder("group_norm_ops")

        b.def_function[group_norm_dispatcher](
            "GroupNorm", docstring="Group normalization"
        )

        return b.finalize()
    except e:
        abort(t"failed to create group_norm op bindings module: {e}")


# =============================================================================
# CPU kernel
# =============================================================================


def _group_norm_cpu[
    dtype: DType,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch_size: Int,
    num_channels: Int,
    spatial_size: Int,
    num_groups: Int,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    """CPU group normalization on [N, C, spatial_size] flattened layout.

    For each batch element and group, computes mean and variance over
    all channels-in-group x spatial elements, then normalizes with
    per-channel gamma and beta.

    Parameters:
        dtype: The data type (must be floating point).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        gamma_ptr: Pointer to the gamma (scale) buffer, shape [C].
        beta_ptr: Pointer to the beta (bias) buffer, shape [C].
        batch_size: Number of samples (N).
        num_channels: Number of channels (C).
        spatial_size: Product of spatial dimensions.
        num_groups: Number of groups to divide channels into.
        epsilon: Small constant for numerical stability.
    """
    var channels_per_group = num_channels // num_groups
    var group_size = channels_per_group * spatial_size

    for n in range(batch_size):
        for g in range(num_groups):
            var c_start = g * channels_per_group

            # Pass 1: compute mean
            var sum_val = Scalar[dtype](0)
            for c_off in range(channels_per_group):
                var c = c_start + c_off
                var base = (n * num_channels + c) * spatial_size
                for s in range(spatial_size):
                    sum_val += in_ptr[base + s]
            var mean_val = sum_val / Scalar[dtype](group_size)

            # Pass 2: compute variance
            var var_val = Scalar[dtype](0)
            for c_off in range(channels_per_group):
                var c = c_start + c_off
                var base = (n * num_channels + c) * spatial_size
                for s in range(spatial_size):
                    var diff = in_ptr[base + s] - mean_val
                    var_val += diff * diff
            var_val = var_val / Scalar[dtype](group_size)

            # Pass 3: normalize with per-channel affine
            var inv_std = Scalar[dtype](1) / sqrt(var_val + epsilon)
            for c_off in range(channels_per_group):
                var c = c_start + c_off
                var base = (n * num_channels + c) * spatial_size
                for s in range(spatial_size):
                    out_ptr[base + s] = (
                        in_ptr[base + s] - mean_val
                    ) * inv_std * gamma_ptr[c] + beta_ptr[c]


# =============================================================================
# GPU kernel wrapper
# =============================================================================


def _group_norm_gpu[
    dtype: DType,
    rank: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    shape: IndexList[rank],
    epsilon: Scalar[dtype],
    num_groups: Int32,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises where dtype.is_floating_point():
    """GPU group normalization via nn.normalization.group_norm.

    Parameters:
        dtype: Element type.
        rank: Input rank (must be 3 or 4).
    """
    comptime if has_accelerator():
        comptime if dtype in (DType.float32, DType.float16, DType.bfloat16):

            @always_inline
            @parameter
            @__copy_capture(in_ptr, shape)
            def input_fn[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
                var c = rebind[IndexList[rank]](coords)
                var flat_idx = 0
                for d in range(rank):
                    flat_idx = flat_idx * shape[d] + c[d]
                return in_ptr.load[width=width](flat_idx)

            @always_inline
            @parameter
            @__copy_capture(gamma_ptr)
            def gamma_fn[
                width: Int
            ](coords: IndexList[1]) -> SIMD[dtype, width]:
                return gamma_ptr.load[width=width](coords[0])

            @always_inline
            @parameter
            @__copy_capture(beta_ptr)
            def beta_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
                return beta_ptr.load[width=width](coords[0])

            comptime out_spec = StaticTensorSpec[dtype, rank, ...].get_unknown()
            var output_tensor = ManagedTensorSlice[
                io_spec=Output, static_spec=out_spec
            ](out_ptr, shape)

            var device_ctx = DeviceContextPtr(ctx.unsafe_value())

            nn_group_norm[
                dtype,
                rank,
                input_fn,
                gamma_fn,
                beta_fn,
                target="gpu",
            ](
                shape,
                epsilon,
                num_groups,
                output_tensor.to_tile_tensor[DType.int64](),
                device_ctx,
            )

        else:
            raise Error(
                "GPU execution not supported for group_norm with dtype "
                + String(dtype)
            )
    else:
        raise Error("No GPU accelerator available")


# =============================================================================
# Dispatcher
# =============================================================================


def _call[
    dtype: DType,
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    gamma_buffer: PythonObject,
    beta_buffer: PythonObject,
    epsilon_buffer: PythonObject,
    num_groups: Int,
    in_shape_py: PythonObject,
    rank: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises where dtype.is_floating_point():
    """Concrete-dtype dispatch helper for group_norm.

    Parameters:
        dtype: Element type (must be floating point).
    """
    var in_shape = _get_shape(in_shape_py, rank)

    var batch_size = in_shape[0]
    var num_channels = in_shape[1]
    var spatial_size = 1
    for i in range(2, rank):
        spatial_size *= in_shape[i]

    var epsilon = _get_buffer_ptr[dtype](epsilon_buffer)[0]

    if not ctx:
        _group_norm_cpu[dtype](
            _get_buffer_ptr[dtype](out_buffer),
            _get_buffer_ptr[dtype](in_buffer),
            _get_buffer_ptr[dtype](gamma_buffer),
            _get_buffer_ptr[dtype](beta_buffer),
            batch_size,
            num_channels,
            spatial_size,
            num_groups,
            epsilon,
        )
    else:
        # GPU path: nn kernel requires rank 3 or 4.
        # For rank 2, reshape to [N, C, 1]. For rank > 4, fall back to error.
        var num_groups_i32 = Int32(num_groups)
        if rank == 2:
            var shape3 = IndexList[3](batch_size, num_channels, 1)
            _group_norm_gpu[dtype, 3](
                _get_buffer_ptr[dtype](out_buffer),
                _get_buffer_ptr[dtype](in_buffer),
                _get_buffer_ptr[dtype](gamma_buffer),
                _get_buffer_ptr[dtype](beta_buffer),
                shape3,
                epsilon,
                num_groups_i32,
                ctx,
            )
        elif rank == 3:
            var shape3 = IndexList[3](batch_size, num_channels, spatial_size)
            _group_norm_gpu[dtype, 3](
                _get_buffer_ptr[dtype](out_buffer),
                _get_buffer_ptr[dtype](in_buffer),
                _get_buffer_ptr[dtype](gamma_buffer),
                _get_buffer_ptr[dtype](beta_buffer),
                shape3,
                epsilon,
                num_groups_i32,
                ctx,
            )
        elif rank == 4:
            var shape4 = IndexList[4](
                in_shape[0], in_shape[1], in_shape[2], in_shape[3]
            )
            _group_norm_gpu[dtype, 4](
                _get_buffer_ptr[dtype](out_buffer),
                _get_buffer_ptr[dtype](in_buffer),
                _get_buffer_ptr[dtype](gamma_buffer),
                _get_buffer_ptr[dtype](beta_buffer),
                shape4,
                epsilon,
                num_groups_i32,
                ctx,
            )
        else:
            raise Error(
                "GPU group_norm only supports rank 2-4, got rank "
                + String(rank)
            )


def group_norm_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    gamma_buffer: PythonObject,
    beta_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Group normalization dispatcher with dtype dispatch.

    Dispatches by dtype. Epsilon and num_groups are read from buffers on CPU.

    Args:
        out_buffer: Output buffer.
        in_buffer: Input buffer.
        gamma_buffer: Gamma (scale) buffer, shape [C].
        beta_buffer: Beta (bias) buffer, shape [C].
        params: Python tuple (epsilon_buffer, num_groups_buffer).
        device_context_ptr: Device context pointer.
    """
    var epsilon_buffer = params[0]
    var num_groups_buffer = params[1]
    var dtype = _get_dtype(in_buffer)
    var ctx = _get_ctx(device_context_ptr)

    var in_shape_py = in_buffer.shape
    var rank = Int(py=len(in_shape_py))

    # num_groups is int32 scalar on CPU
    var num_groups = Int(_get_buffer_ptr[DType.int32](num_groups_buffer)[0])

    if dtype == DType.float16:
        _call[DType.float16](
            out_buffer,
            in_buffer,
            gamma_buffer,
            beta_buffer,
            epsilon_buffer,
            num_groups,
            in_shape_py,
            rank,
            ctx,
        )
    elif dtype == DType.float32:
        _call[DType.float32](
            out_buffer,
            in_buffer,
            gamma_buffer,
            beta_buffer,
            epsilon_buffer,
            num_groups,
            in_shape_py,
            rank,
            ctx,
        )
    elif dtype == DType.float64:
        _call[DType.float64](
            out_buffer,
            in_buffer,
            gamma_buffer,
            beta_buffer,
            epsilon_buffer,
            num_groups,
            in_shape_py,
            rank,
            ctx,
        )
    elif dtype == DType.bfloat16:
        _call[DType.bfloat16](
            out_buffer,
            in_buffer,
            gamma_buffer,
            beta_buffer,
            epsilon_buffer,
            num_groups,
            in_shape_py,
            rank,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for group_norm: " + String(dtype))
