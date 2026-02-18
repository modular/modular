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
"""RMSNorm with fused residual connection for state space models."""

from math import align_down, ceildiv, rsqrt
from sys.info import align_of, simd_width_of, size_of

from algorithm import vectorize
from algorithm.functional import _get_start_indices_of_nth_subvolume
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from gpu.host.info import is_gpu
from gpu.memory import external_memory
from gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from layout import IntTuple, Layout, LayoutTensor, RuntimeTuple, UNKNOWN_VALUE
from layout._coord import Coord, CoordLike, Idx
from layout._layout import Layout as TileLayout, row_major
from layout._tile_tensor import TileTensor
from random import Random
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type

from nn.normalization import _rms_norm_gpu_block_subkernel, _sum_to_mean


# ===----------------------------------------------------------------------=== #
# CPU Implementations
# ===----------------------------------------------------------------------=== #


fn _rms_norm_fused_residual_cpu_2d[
    dtype: DType,
    //,
    input_fn: fn[width: Int](Int, Int) capturing -> SIMD[dtype, width],
    residual_input_fn: fn[width: Int](Int, Int) capturing -> SIMD[dtype, width],
    output_fn: fn[width: Int, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) capturing -> None,
    residual_read_fn: fn[width: Int](Int, Int) capturing -> SIMD[dtype, width],
    multiply_before_cast: Bool = True,
](
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    out_shape: IndexList[2],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
):
    """Core 2D implementation of RMSNorm with fused residual.

    Uses simple (row, col) indexing to avoid compile-time evaluation issues.
    """
    comptime assert gamma.rank == 1, "gamma must have rank 1"

    var num_rows = out_shape[0]
    var num_cols = out_shape[1]

    comptime simd_width = simd_width_of[dtype]()
    var simd_loop_end = align_down(num_cols, simd_width)
    comptime intermediate_type = get_accum_type[dtype]()

    # Calculate dropout scale if needed
    var dropout_scale = Scalar[dtype](1.0)
    var zero_scalar = Scalar[dtype](0.0)
    if dropout_p > zero_scalar:
        var one_scalar = Scalar[dtype](1.0)
        dropout_scale = one_scalar / (one_scalar - dropout_p)

    for var row in range(num_rows):
        # First compute sum of squared (input + residual) for RMSNorm
        var sum_simd = SIMD[intermediate_type, simd_width]()

        # SIMD loop
        for col in range(0, simd_loop_end, simd_width):
            var input_vals = input_fn[simd_width](row, col)
            var residual_vals = residual_input_fn[simd_width](row, col)

            # Apply dropout if enabled
            if dropout_p > zero_scalar:

                @parameter
                for i in range(simd_width):
                    var element_offset = row * num_cols + col + i
                    var generator = Random(
                        seed=seed, offset=UInt64(element_offset)
                    )
                    var rng = generator.step_uniform()
                    var rng_val = rng[0].cast[dtype]()
                    if rng_val >= dropout_p:
                        input_vals[i] = input_vals[i] * dropout_scale
                    else:
                        input_vals[i] = zero_scalar

            var sum_vals = input_vals + residual_vals

            # Output pre-normalized value (x + residual)
            output_residual_fn[simd_width, 1](row, col, sum_vals)

            # Accumulate for RMSNorm
            sum_simd += sum_vals.cast[intermediate_type]() ** 2

        # Scalar loop for remainder
        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            var input_val = input_fn[1](row, col)[0]
            var residual_val = residual_input_fn[1](row, col)[0]

            # Apply dropout if enabled
            if dropout_p > zero_scalar:
                var element_offset = row * num_cols + col
                var generator = Random(seed=seed, offset=UInt64(element_offset))
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p:
                    input_val = input_val * dropout_scale
                else:
                    input_val = zero_scalar

            var sum_val_scalar = input_val + residual_val

            # Output pre-normalized value
            output_residual_fn[1, 1](row, col, sum_val_scalar)

            # Accumulate for RMSNorm
            sum_val += sum_val_scalar.cast[intermediate_type]() ** 2

        # Compute normalization factor
        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = rsqrt(mean_val + epsilon.cast[intermediate_type]())

        # Second pass: apply normalization
        fn _normalize[sw: Int](col: Int) unified {mut}:
            # Read the pre-computed sum values (input + residual) from first pass
            var sum_vals = residual_read_fn[sw](row, col).cast[
                intermediate_type
            ]()

            var gamma_col = gamma.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](col)
            )
            var gamma_val = gamma.ptr.load[width=sw](gamma_col)
            var norm_val: SIMD[dtype, sw]

            if multiply_before_cast:
                var gamma_offset = gamma_val + weight_offset
                norm_val = (sum_vals * norm_factor).cast[dtype]() * gamma_offset
            else:
                norm_val = (sum_vals * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset
                )

            output_fn[sw, 1](row, col, norm_val)

        vectorize[simd_width](num_cols, _normalize)


fn rms_norm_fused_residual_cpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_fn: fn[width: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    residual_read_fn: fn[width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    /,
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Generic rank wrapper that delegates to the 2D core implementation.

    Creates 2D wrapper lambdas that translate (row, col) to IndexList[rank]
    at runtime, avoiding compile-time evaluation issues with _lambda_load.
    """
    comptime assert gamma.rank == 1, "gamma must have rank 1"

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    # Create 2D wrapper lambdas that translate indices at runtime
    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width, rank](indices)

    @parameter
    @always_inline
    fn residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width, rank](indices)

    @parameter
    @always_inline
    fn output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices, val)

    @parameter
    @always_inline
    fn output_residual_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices, val)

    @parameter
    @always_inline
    fn residual_read_fn_2d[sw: Int](row: Int, col: Int) -> SIMD[dtype, sw]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_read_fn[sw, rank](indices)

    # Call the 2D core implementation
    _rms_norm_fused_residual_cpu_2d[
        input_fn_2d,
        residual_input_fn_2d,
        output_fn_2d,
        output_residual_fn_2d,
        residual_read_fn_2d,
        multiply_before_cast=multiply_before_cast,
    ](
        gamma,
        epsilon,
        weight_offset,
        out_shape=IndexList[2](prod_all_but_last_dim, last_dim),
        dropout_p=dropout_p,
        seed=seed,
    )


# ===----------------------------------------------------------------------=== #
# GPU Implementations
# ===----------------------------------------------------------------------=== #


fn rms_norm_fused_residual_gpu_block[
    mut: Bool,
    origin: Origin[mut=mut],
    layout: Layout,
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
):
    comptime assert gamma.rank == 1, "gamma must have rank 1"

    var shared_mem = external_memory[
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
        alignment = align_of[SIMD[dtype, simd_width]](),
        name="intermediate_shared_memory",
    ]()
    with PDL():
        # First stage: apply dropout, add residual to input and store in shared memory.
        # Loop to handle cases where num_cols > block_dim * simd_width,
        # matching the loop structure in _rms_norm_gpu_block_subkernel.
        var tid = thread_idx.x
        var row = block_idx.x

        for x in range(ceildiv(num_cols // simd_width, Int(block_dim.x))):
            var idx = x * Int(block_dim.x) * simd_width + Int(
                tid * UInt(simd_width)
            )

            if idx < num_cols:
                var input_val = input_fn[simd_width](Int(row), idx)

                # Apply dropout if enabled
                var zero_scalar = Scalar[dtype](0.0)
                if dropout_p > zero_scalar:
                    var one_scalar = Scalar[dtype](1.0)
                    var dropout_scale = one_scalar / (one_scalar - dropout_p)

                    for i in range(simd_width):
                        if idx + i < num_cols:
                            # Use element position as offset for RNG to ensure different values per element
                            var element_offset = (
                                UInt64(row) * UInt64(num_cols)
                                + UInt64(idx)
                                + UInt64(i)
                            )
                            var generator = Random(
                                seed=seed, offset=element_offset
                            )
                            var rng = generator.step_uniform()
                            var rng_val = rng[0].cast[dtype]()
                            if rng_val >= dropout_p:
                                input_val[i] = input_val[i] * dropout_scale
                            else:
                                input_val[i] = zero_scalar

                var residual_val = residual_input_fn[simd_width](Int(row), idx)
                var residual_add_val = input_val + residual_val

                # Output the pre-normalized value (x + residual) for prenorm mode
                output_residual_fn[
                    simd_width, align_of[SIMD[dtype, simd_width]]()
                ](Int(row), idx, residual_add_val)

                # Store in shared memory for normalization
                shared_mem.store[
                    width=simd_width,
                    alignment = align_of[SIMD[dtype, simd_width]](),
                ](idx, residual_add_val)

        barrier()

        # Second stage: apply RMSNorm using shared memory as input
        @parameter
        @always_inline
        @__copy_capture(shared_mem)
        fn shared_mem_input_fn[
            width: Int
        ](row: Int, col: Int) -> SIMD[dtype, width]:
            return shared_mem.load[width=width](col)

        # Construct a TileTensor from the LayoutTensor's pointer for the subkernel
        var gamma_tile = TileTensor(gamma.ptr, row_major(Coord(Idx(num_cols))))

        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            shared_mem_input_fn,
            output_fn,
            multiply_before_cast,
        ](gamma_tile, epsilon, weight_offset, num_cols)


fn rms_norm_fused_residual_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_residual_fn: fn[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_fn: fn[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank, **_],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    comptime assert gamma.rank == 1, "gamma must have rank 1"

    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    var rows = shape.flattened_length() // last_dim
    var cols = last_dim

    @parameter
    @always_inline
    fn output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    fn output_residual_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @parameter
    @always_inline
    fn residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width](indices.canonicalize())

    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    var shared_mem_size = (
        ceildiv(cols, simd_width) * simd_width * size_of[dtype]()
    )

    comptime kernel = rms_norm_fused_residual_gpu_block[
        mut = gamma.mut,
        origin = gamma.origin,
        layout = gamma.layout,
        simd_width,
        max_warps_per_block,
        input_fn_2d,
        residual_input_fn_2d,
        output_fn_2d,
        output_residual_fn_2d,
        multiply_before_cast=multiply_before_cast,
    ]
    ctx.enqueue_function[kernel, kernel](
        gamma,
        epsilon,
        weight_offset,
        cols,
        dropout_p,
        seed,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(),
        shared_mem_bytes=shared_mem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(
                ctx.default_device_info.shared_memory_per_multiprocessor - 4096
            )
        ),
    )


fn _rms_norm_fused_residual_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContextPtr,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    comptime assert gamma.rank == 1, "gamma must have rank 1"

    # Note: we only support reduction along the last dimension
    if gamma.runtime_layout.shape.value[0] != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.runtime_layout.shape.value[0])
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    @parameter
    if is_gpu[target]():
        rms_norm_fused_residual_gpu[
            input_0_fn,
            input_1_fn,
            output_residual_fn,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx.get_device_context(),
            dropout_p,
            seed,
        )
    else:
        # Create a read function that re-computes the sum (matching first pass)
        # This is needed because we don't have direct access to the residual output buffer
        @__copy_capture(dropout_p, seed, shape)
        @parameter
        @always_inline
        fn residual_read_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            var input_vals = input_0_fn[width, _rank](coords)
            var residual_vals = input_1_fn[width, _rank](coords)

            # Apply dropout if enabled (matching first pass exactly)
            var zero_scalar = Scalar[dtype](0.0)
            if dropout_p > zero_scalar:
                var one_scalar = Scalar[dtype](1.0)
                var dropout_scale = one_scalar / (one_scalar - dropout_p)
                var last_dim = shape[_rank - 1]
                var row = coords.flattened_length() // last_dim

                @parameter
                for i in range(width):
                    var col_idx = coords[_rank - 1] + i
                    var element_offset = row * last_dim + col_idx
                    var generator = Random(
                        seed=seed, offset=UInt64(element_offset)
                    )
                    var rng = generator.step_uniform()
                    var rng_val = rng[0].cast[dtype]()
                    if rng_val >= dropout_p:
                        input_vals[i] = input_vals[i] * dropout_scale
                    else:
                        input_vals[i] = zero_scalar

            return input_vals + residual_vals

        rms_norm_fused_residual_cpu[
            input_0_fn,
            input_1_fn,
            output_fn,
            output_residual_fn,
            residual_read_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            dropout_p,
            seed,
        )


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


@always_inline
fn rms_norm_fused_residual[
    dtype: DType,
    rank: Int,
    //,
    input_0_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_0_fn: fn[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContextPtr,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    comptime assert gamma.rank == 1, "gamma must have rank 1"

    @always_inline
    @parameter
    fn output_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    fn output_residual_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_residual_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_residual",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        _rms_norm_fused_residual_impl[
            dtype,
            rank,
            input_0_fn,
            input_1_fn,
            output_fn_wrapper,
            output_residual_fn_wrapper,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx,
            dropout_p,
            seed,
        )
