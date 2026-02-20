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

from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from utils.index import IndexList
from memory import UnsafePointer
from algorithm import sync_parallelize
import math
from math import ceildiv, exp, exp2, rsqrt
from state_space.causal_conv1d import silu

# ===----------------------------------------------------------------------=== #
# Constants and Type Aliases
# ===----------------------------------------------------------------------=== #

comptime LOG2E = 1.4426950408889634  # For converting exp to exp2 (faster on GPU)
comptime MAX_DSTATE = 16

# Stride types for passing tensor strides to kernels
comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


# ===----------------------------------------------------------------------=== #
# Activation Functions
# ===----------------------------------------------------------------------=== #


fn softplus(val: Float32) -> Float32:
    """Softplus activation: log(1 + exp(x)) with numerical stability."""
    if val > 20.0:
        return val
    var exp_val = math.exp(val)
    return math.log(1.0 + exp_val)


fn sigmoid(val: Float32) -> Float32:
    """Optimized sigmoid using fast approximation for large negative values."""
    if val < -20.0:
        return 0.0
    var exp_neg = math.exp(-val)
    return 1.0 / (1.0 + exp_neg)


# ===----------------------------------------------------------------------=== #
# Selective Scan Forward Kernels (Full Sequence)
# ===----------------------------------------------------------------------=== #


fn selective_scan_fwd_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    output_strides: Strides3D,
    x_strides: Strides4D,
    out_z_strides: Strides3D,
    u_strides: Strides3D,
    delta_strides: Strides3D,
    A_strides: Strides2D,
    B_strides: Strides4D,
    C_strides: Strides4D,
    D_strides: Strides1D,
    z_strides: Strides3D,
    delta_bias_strides: Strides1D,
):
    """GPU kernel for selective scan forward pass.

    Each thread processes one (batch, dim) pair and iterates through the sequence.
    """
    # Calculate which (batch, dim) this thread is responsible for
    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return

    var b = thread_id_int // dim
    var d = thread_id_int % dim

    # Additional bounds checking
    if b >= batch or d >= dim:
        return

    var group_id = d // group_size

    # Local state storage (max dstate 16 to fit in registers)
    # Note: Using large SIMD sizes (e.g. 256) causes register spilling and massive performance loss
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
    var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

    # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
    # This optimization converts exp(A * delta) to exp2(A * LOG2E * delta)
    # which is faster on GPUs
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    var has_delta_bias = delta_bias.dim(0) > 0
    var delta_bias_val = Float32(0.0)
    if has_delta_bias:
        var bias_offset = UInt32(d * delta_bias_strides[0])
        delta_bias_val = Scalar[kernel_dtype](delta_bias.ptr[bias_offset]).cast[
            DType.float32
        ]()

    var has_D = D.dim(0) > 0
    var D_val = Float32(0.0)
    if has_D:
        var D_offset = UInt32(d * D_strides[0])
        D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    var has_z = z.dim(0) > 0
    var has_out_z = out_z.dim(0) > 0

    # Pre-multiply A by LOG2E for exp2 optimization
    comptime for n in range(DSTATE):
        var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
        A_vals[n] = (
            Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]() * LOG2E
        )

    var chunk_size = 2048
    var t_in_chunk = 0
    var chunk_idx = 0

    # Initialize running offsets for strength reduction
    var curr_u_offset = UInt32(b * u_strides[0] + d * u_strides[1])
    var curr_delta_offset = UInt32(b * delta_strides[0] + d * delta_strides[1])
    var curr_output_offset = UInt32(
        b * output_strides[0] + d * output_strides[1]
    )
    var curr_B_offset = UInt32(b * B_strides[0] + group_id * B_strides[1])
    var curr_C_offset = UInt32(b * C_strides[0] + group_id * C_strides[1])
    var curr_z_offset = UInt32(b * z_strides[0] + d * z_strides[1])
    var curr_out_z_offset = UInt32(b * out_z_strides[0] + d * out_z_strides[1])

    # Process sequence sequentially for this (batch, dim)
    # OPTIMIZED: Tiled loading with pre-loaded B/C tiles and buffered outputs
    comptime TILE_SIZE = 8  # Sweet spot: larger causes register spilling
    var aligned_seqlen = seqlen - (seqlen % TILE_SIZE)
    var t = 0

    # Check if we can use contiguous loads (stride == 1)
    # Note: u tensor always uses strided access to handle different layouts from causal_conv1d_fn
    var delta_contiguous = delta_strides[2] == 1
    var z_contiguous = z_strides[2] == 1
    var B_contiguous = B_strides[3] == 1
    var C_contiguous = C_strides[3] == 1
    var output_contiguous = output_strides[2] == 1

    # Fast path: Tiled loading for the aligned portion
    while t < aligned_seqlen:
        # Load u and delta vectors
        var u_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var delta_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var z_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)

        # Always use strided access for u tensor to handle different layouts
        # from causal_conv1d_fn and other operations
        for i in range(TILE_SIZE):
            u_vec[i] = u.ptr[curr_u_offset + UInt32(i * u_strides[2])]

        # Always use strided access for delta tensor to handle different layouts
        for i in range(TILE_SIZE):
            delta_vec[i] = delta.ptr[
                curr_delta_offset + UInt32(i * delta_strides[2])
            ]

        if has_z:
            # Always use strided access for z tensor to handle different layouts
            for i in range(TILE_SIZE):
                z_vec[i] = z.ptr[curr_z_offset + UInt32(i * z_strides[2])]

        # PRE-LOAD B/C TILES: Load B[n, t:t+TILE] and C[n, t:t+TILE] for all n
        # This avoids redundant address calculations inside the inner loop
        var B_tiles = InlineArray[SIMD[DType.float32, TILE_SIZE], DSTATE](
            fill=0
        )
        var C_tiles = InlineArray[SIMD[DType.float32, TILE_SIZE], DSTATE](
            fill=0
        )

        # Load B tiles - always use scalar loads to handle different layouts from slicing/reshaping
        for i in range(TILE_SIZE):
            var b_base = curr_B_offset + UInt32(i * B_strides[3])

            comptime for n in range(DSTATE):
                B_tiles[n][i] = Scalar[kernel_dtype](
                    B.ptr[b_base + UInt32(n * B_strides[2])]
                ).cast[DType.float32]()

        # Load C tiles - always use scalar loads to handle different layouts from slicing/reshaping
        for i in range(TILE_SIZE):
            var c_base = curr_C_offset + UInt32(i * C_strides[3])

            comptime for n in range(DSTATE):
                C_tiles[n][i] = Scalar[kernel_dtype](
                    C.ptr[c_base + UInt32(n * C_strides[2])]
                ).cast[DType.float32]()

        # Buffer for output values to enable vector stores
        var output_buffer = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var out_z_buffer = SIMD[kernel_dtype, TILE_SIZE](0.0)

        # Process tile with pre-loaded B/C values
        for i in range(TILE_SIZE):
            t_in_chunk += 1

            # Extract scalars from pre-loaded vectors
            var u_val = u_vec[i].cast[DType.float32]()
            var delta_val = delta_vec[i].cast[DType.float32]()

            # Apply delta bias and softplus
            if has_delta_bias:
                delta_val += delta_bias_val
            if delta_softplus_bool:
                delta_val = softplus(delta_val)

            var delta_u = delta_val * u_val

            # Extract B/C values for this timestep from pre-loaded tiles
            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

            comptime for n in range(DSTATE):
                B_vals[n] = B_tiles[n][i]
                C_vals[n] = C_tiles[n][i]

            # SIMD Math
            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var output_val = (state * C_vals).reduce_add()

            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t

            if has_D:
                output_val += D_val * u_val

            # Buffer output for vector store
            output_buffer[i] = output_val.cast[kernel_dtype]()

            if has_z:
                var z_val = z_vec[i].cast[DType.float32]()
                var out_z_val = output_val * silu(z_val)
                out_z_buffer[i] = out_z_val.cast[kernel_dtype]()

            # Checkpoint handling
            var current_t = t + i
            var is_chunk_boundary = t_in_chunk == chunk_size
            var is_last_step = current_t == seqlen - 1

            if is_chunk_boundary or is_last_step:
                comptime for n in range(DSTATE):
                    var x_offset_a = UInt32(
                        b * x_strides[0]
                        + d * x_strides[1]
                        + chunk_idx * x_strides[2]
                        + (n * 2) * x_strides[3]
                    )
                    var x_offset_b = UInt32(
                        b * x_strides[0]
                        + d * x_strides[1]
                        + chunk_idx * x_strides[2]
                        + (n * 2 + 1) * x_strides[3]
                    )
                    x.ptr[x_offset_a] = Scalar[kernel_dtype](
                        cum_a[n].cast[kernel_dtype]()
                    )
                    x.ptr[x_offset_b] = Scalar[kernel_dtype](
                        cum_b[n].cast[kernel_dtype]()
                    )
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0

                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0

        # Vector store outputs if contiguous
        if output_contiguous:
            for i in range(TILE_SIZE):
                output.ptr[curr_output_offset + UInt32(i)] = output_buffer[i]
        else:
            for i in range(TILE_SIZE):
                output.ptr[
                    curr_output_offset + UInt32(i * output_strides[2])
                ] = output_buffer[i]

        if has_z and has_out_z:
            if out_z_strides[2] == 1:
                for i in range(TILE_SIZE):
                    out_z.ptr[curr_out_z_offset + UInt32(i)] = out_z_buffer[i]
            else:
                for i in range(TILE_SIZE):
                    out_z.ptr[
                        curr_out_z_offset + UInt32(i * out_z_strides[2])
                    ] = out_z_buffer[i]

        # Advance global offsets by TILE_SIZE
        curr_u_offset += UInt32(u_strides[2] * TILE_SIZE)
        curr_delta_offset += UInt32(delta_strides[2] * TILE_SIZE)
        curr_output_offset += UInt32(output_strides[2] * TILE_SIZE)
        curr_B_offset += UInt32(B_strides[3] * TILE_SIZE)
        curr_C_offset += UInt32(C_strides[3] * TILE_SIZE)
        curr_z_offset += UInt32(z_strides[2] * TILE_SIZE)
        curr_out_z_offset += UInt32(out_z_strides[2] * TILE_SIZE)

        t += TILE_SIZE

    # Tail loop (scalar)
    while t < seqlen:
        t_in_chunk += 1
        var u_val = Scalar[kernel_dtype](u.ptr[curr_u_offset]).cast[
            DType.float32
        ]()
        var delta_val = Scalar[kernel_dtype](delta.ptr[curr_delta_offset]).cast[
            DType.float32
        ]()
        if has_delta_bias:
            delta_val += delta_bias_val
        if delta_softplus_bool:
            delta_val = softplus(delta_val)
        var delta_u = delta_val * u_val
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.ptr[curr_B_offset + UInt32(n * B_strides[2])]
            ).cast[DType.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.ptr[curr_C_offset + UInt32(n * C_strides[2])]
            ).cast[DType.float32]()
        var a_t = exp2(A_vals * delta_val)
        var b_t = B_vals * delta_u
        state = state * a_t + b_t
        var output_val = (state * C_vals).reduce_add()
        cum_b = cum_b * a_t + b_t
        cum_a = cum_a * a_t
        if has_D:
            output_val += D_val * u_val
        output.ptr[curr_output_offset] = Scalar[kernel_dtype](
            output_val.cast[kernel_dtype]()
        )
        if has_z:
            var z_val = Scalar[kernel_dtype](z.ptr[curr_z_offset]).cast[
                DType.float32
            ]()
            var out_z_val = output_val * silu(z_val)
            if has_out_z:
                out_z.ptr[curr_out_z_offset] = Scalar[kernel_dtype](
                    out_z_val.cast[kernel_dtype]()
                )

        curr_u_offset += UInt32(u_strides[2])
        curr_delta_offset += UInt32(delta_strides[2])
        curr_output_offset += UInt32(output_strides[2])
        curr_B_offset += UInt32(B_strides[3])
        curr_C_offset += UInt32(C_strides[3])
        curr_z_offset += UInt32(z_strides[2])
        curr_out_z_offset += UInt32(out_z_strides[2])

        var is_chunk_boundary = t_in_chunk == chunk_size
        var is_last_step = t == seqlen - 1
        if is_chunk_boundary or is_last_step:
            comptime for n in range(DSTATE):
                var x_offset_a = UInt32(
                    b * x_strides[0]
                    + d * x_strides[1]
                    + chunk_idx * x_strides[2]
                    + (n * 2) * x_strides[3]
                )
                var x_offset_b = UInt32(
                    b * x_strides[0]
                    + d * x_strides[1]
                    + chunk_idx * x_strides[2]
                    + (n * 2 + 1) * x_strides[3]
                )
                x.ptr[x_offset_a] = Scalar[kernel_dtype](
                    cum_a[n].cast[kernel_dtype]()
                )
                x.ptr[x_offset_b] = Scalar[kernel_dtype](
                    cum_b[n].cast[kernel_dtype]()
                )
                cum_a[n] = 1.0
                cum_b[n] = 0.0
            if is_chunk_boundary:
                chunk_idx += 1
                t_in_chunk = 0
        t += 1


fn selective_scan_fwd_gpu_minimal[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    x_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    output_strides: Strides3D,
    x_strides: Strides4D,
    u_strides: Strides3D,
    delta_strides: Strides3D,
    A_strides: Strides2D,
    B_strides: Strides4D,
    C_strides: Strides4D,
):
    """Minimal GPU kernel for selective scan forward - no D, z, or delta_bias.
    """
    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return

    var b = thread_id_int // dim
    var d = thread_id_int % dim

    if b >= batch or d >= dim:
        return

    var group_id = d // group_size

    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
    var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)

    comptime for n in range(DSTATE):
        var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
        A_vals[n] = (
            Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]() * LOG2E
        )

    var chunk_size = 2048
    var t_in_chunk = 0
    var chunk_idx = 0

    var curr_u_offset = UInt32(b * u_strides[0] + d * u_strides[1])
    var curr_delta_offset = UInt32(b * delta_strides[0] + d * delta_strides[1])
    var curr_output_offset = UInt32(
        b * output_strides[0] + d * output_strides[1]
    )
    var curr_B_offset = UInt32(b * B_strides[0] + group_id * B_strides[1])
    var curr_C_offset = UInt32(b * C_strides[0] + group_id * C_strides[1])

    # Simple scalar loop - no tiling for simplicity in minimal version
    for t in range(seqlen):
        t_in_chunk += 1

        var u_val = Scalar[kernel_dtype](u.ptr[curr_u_offset]).cast[
            DType.float32
        ]()
        var delta_val = Scalar[kernel_dtype](delta.ptr[curr_delta_offset]).cast[
            DType.float32
        ]()
        if delta_softplus_bool:
            delta_val = softplus(delta_val)
        var delta_u = delta_val * u_val

        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.ptr[curr_B_offset + UInt32(n * B_strides[2])]
            ).cast[DType.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.ptr[curr_C_offset + UInt32(n * C_strides[2])]
            ).cast[DType.float32]()

        var a_t = exp2(A_vals * delta_val)
        var b_t = B_vals * delta_u
        state = state * a_t + b_t
        var output_val = (state * C_vals).reduce_add()
        cum_b = cum_b * a_t + b_t
        cum_a = cum_a * a_t
        # No D skip connection in minimal version
        output.ptr[curr_output_offset] = Scalar[kernel_dtype](
            output_val.cast[kernel_dtype]()
        )
        # No z gating in minimal version

        curr_u_offset += UInt32(u_strides[2])
        curr_delta_offset += UInt32(delta_strides[2])
        curr_output_offset += UInt32(output_strides[2])
        curr_B_offset += UInt32(B_strides[3])
        curr_C_offset += UInt32(C_strides[3])

        var is_chunk_boundary = t_in_chunk == chunk_size
        var is_last_step = t == seqlen - 1
        if is_chunk_boundary or is_last_step:
            comptime for n in range(DSTATE):
                var x_offset_a = UInt32(
                    b * x_strides[0]
                    + d * x_strides[1]
                    + chunk_idx * x_strides[2]
                    + (n * 2) * x_strides[3]
                )
                var x_offset_b = UInt32(
                    b * x_strides[0]
                    + d * x_strides[1]
                    + chunk_idx * x_strides[2]
                    + (n * 2 + 1) * x_strides[3]
                )
                x.ptr[x_offset_a] = Scalar[kernel_dtype](
                    cum_a[n].cast[kernel_dtype]()
                )
                x.ptr[x_offset_b] = Scalar[kernel_dtype](
                    cum_b[n].cast[kernel_dtype]()
                )
                cum_a[n] = 1.0
                cum_b[n] = 0.0
            if is_chunk_boundary:
                chunk_idx += 1
                t_in_chunk = 0


# ===----------------------------------------------------------------------=== #
# Selective Scan Update Kernel (Single Step / Autoregressive)
# ===----------------------------------------------------------------------=== #
# This kernel is used for incremental/autoregressive inference where you
# process one token at a time and update the hidden state.
#
# Reference: mamba_ssm/ops/triton/selective_state_update.py
#
# Algorithm:
#   dt = dt + dt_bias (if has_dt_bias)
#   dt = softplus(dt) (if dt_softplus)
#   dA = exp(A * dt)
#   dB = B * dt
#   state = state * dA + dB * x
#   out = sum(state * C, axis=-1)
#   out += x * D (if has_D)
#   out *= z * sigmoid(z) (if has_z)
# ===----------------------------------------------------------------------=== #


fn selective_scan_update_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    state_out_layout: Layout,
    output_layout: Layout,
    state_in_layout: Layout,
    x_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    dt_bias_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    dim: Int,
    group_size: Int,
    delta_softplus: Int8,
    state_out: LayoutTensor[kernel_dtype, state_out_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    state_in: LayoutTensor[kernel_dtype, state_in_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, dt_bias_layout, MutAnyOrigin],
    state_out_strides: Strides3D,
    output_strides: Strides2D,
    state_in_strides: Strides3D,
    x_strides: Strides2D,
    dt_strides: Strides2D,
    A_strides: Strides2D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    z_strides: Strides2D,
    dt_bias_strides: Strides1D,
):
    """GPU kernel for selective scan update (single step).

    Each thread processes one (batch, dim) pair.
    Reads initial state from state_in, writes updated state to state_out.
    """
    # Calculate which (batch, dim) this thread is responsible for
    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return

    var b = thread_id_int // dim
    var d = thread_id_int % dim

    # Additional bounds checking
    if b >= batch or d >= dim:
        return

    # Compute group_id for this dimension
    var group_id = d // group_size

    # Load dt value
    var dt_offset = UInt32(b * dt_strides[0] + d * dt_strides[1])
    var dt_val = Scalar[kernel_dtype](dt.ptr[dt_offset]).cast[DType.float32]()

    # Apply dt_bias if present
    var has_dt_bias = dt_bias.dim(0) > 0
    if has_dt_bias:
        var bias_offset = UInt32(d * dt_bias_strides[0])
        var bias_val = Scalar[kernel_dtype](dt_bias.ptr[bias_offset]).cast[
            DType.float32
        ]()
        dt_val += bias_val

    # Apply softplus if requested
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    if delta_softplus_bool:
        dt_val = softplus(dt_val)

    # Load x value
    var x_offset = UInt32(b * x_strides[0] + d * x_strides[1])
    var x_val = Scalar[kernel_dtype](x.ptr[x_offset]).cast[DType.float32]()

    # Load A values for this dim and pre-multiply by LOG2E for faster exp2
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
        A_vals[n] = (
            Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]() * LOG2E
        )

    # Compute dA = exp2(A * LOG2E * dt) = exp(A * dt)
    var dA = exp2(A_vals * dt_val)

    # Load B values using group_id
    var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var B_offset = UInt32(
            b * B_strides[0] + group_id * B_strides[1] + n * B_strides[2]
        )
        B_vals[n] = Scalar[kernel_dtype](B.ptr[B_offset]).cast[DType.float32]()

    # Compute dB = B * dt
    var dB = B_vals * dt_val

    # Load current state from state_in
    var state_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var state_offset = UInt32(
            b * state_in_strides[0]
            + d * state_in_strides[1]
            + n * state_in_strides[2]
        )
        state_vals[n] = Scalar[kernel_dtype](state_in.ptr[state_offset]).cast[
            DType.float32
        ]()

    # Update state: state = state * dA + dB * x
    state_vals = state_vals * dA + dB * x_val

    # Store updated state to state_out
    comptime for n in range(DSTATE):
        var state_offset = UInt32(
            b * state_out_strides[0]
            + d * state_out_strides[1]
            + n * state_out_strides[2]
        )
        state_out.ptr[state_offset] = Scalar[kernel_dtype](
            state_vals[n].cast[kernel_dtype]()
        )
    # Load C values using group_id
    var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var C_offset = UInt32(
            b * C_strides[0] + group_id * C_strides[1] + n * C_strides[2]
        )
        C_vals[n] = Scalar[kernel_dtype](C.ptr[C_offset]).cast[DType.float32]()

    # Compute output: out = sum(state * C, axis=-1)
    var out_val = (state_vals * C_vals).reduce_add()

    # Add skip connection if D is present
    var has_D = D.dim(0) > 0
    if has_D:
        var D_offset = UInt32(d * D_strides[0])
        var D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()
        out_val += x_val * D_val

    # Apply gating if z is present
    var has_z = z.dim(0) > 0
    if has_z:
        var z_offset = UInt32(b * z_strides[0] + d * z_strides[1])
        var z_val = Scalar[kernel_dtype](z.ptr[z_offset]).cast[DType.float32]()
        out_val *= z_val * sigmoid(
            z_val
        )  # z * sigmoid(z) = silu(z) but formulated differently

    # Store output
    var out_offset = UInt32(b * output_strides[0] + d * output_strides[1])
    output.ptr[out_offset] = Scalar[kernel_dtype](out_val.cast[kernel_dtype]())


fn selective_scan_update_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
    state_out_layout: Layout,
    output_layout: Layout,
    state_in_layout: Layout,
    x_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    dt_bias_layout: Layout,
](
    batch: Int,
    dim: Int,
    group_size: Int,
    delta_softplus: Int8,
    state_out: LayoutTensor[kernel_dtype, state_out_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    state_in: LayoutTensor[kernel_dtype, state_in_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, dt_bias_layout, MutAnyOrigin],
    state_out_strides: Strides3D,
    output_strides: Strides2D,
    state_in_strides: Strides3D,
    x_strides: Strides2D,
    dt_strides: Strides2D,
    A_strides: Strides2D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    z_strides: Strides2D,
    dt_bias_strides: Strides1D,
):
    """CPU kernel for selective scan update (single step)."""
    var has_dt_bias = dt_bias.dim(0) > 0
    var has_D = D.dim(0) > 0
    var has_z = z.dim(0) > 0
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)

    @parameter
    fn worker(idx: Int):
        var b = idx // dim
        var d = idx % dim

        # Compute group_id for this dimension
        var group_id = d // group_size

        # Load dt value
        var dt_offset = UInt32(b * dt_strides[0] + d * dt_strides[1])
        var dt_val = Scalar[kernel_dtype](dt.ptr[dt_offset]).cast[
            DType.float32
        ]()

        # Apply dt_bias if present
        if has_dt_bias:
            var bias_offset = UInt32(d * dt_bias_strides[0])
            var bias_val = Scalar[kernel_dtype](dt_bias.ptr[bias_offset]).cast[
                DType.float32
            ]()
            dt_val += bias_val

        # Apply softplus if requested
        if delta_softplus_bool:
            dt_val = softplus(dt_val)

        # Load x value
        var x_offset = UInt32(b * x_strides[0] + d * x_strides[1])
        var x_val = Scalar[kernel_dtype](x.ptr[x_offset]).cast[DType.float32]()

        # Load A values and pre-multiply by LOG2E
        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
            A_vals[n] = (
                Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]()
                * LOG2E
            )

        # Compute dA
        var dA = exp2(A_vals * dt_val)

        # Load B values using group_id
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var B_offset = UInt32(
                b * B_strides[0] + group_id * B_strides[1] + n * B_strides[2]
            )
            B_vals[n] = Scalar[kernel_dtype](B.ptr[B_offset]).cast[
                DType.float32
            ]()

        # Compute dB
        var dB = B_vals * dt_val

        # Load current state from state_in
        var state_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var state_offset = UInt32(
                b * state_in_strides[0]
                + d * state_in_strides[1]
                + n * state_in_strides[2]
            )
            state_vals[n] = Scalar[kernel_dtype](
                state_in.ptr[state_offset]
            ).cast[DType.float32]()

        # Update state
        state_vals = state_vals * dA + dB * x_val

        # Store updated state to state_out
        comptime for n in range(DSTATE):
            var state_offset = UInt32(
                b * state_out_strides[0]
                + d * state_out_strides[1]
                + n * state_out_strides[2]
            )
            state_out.ptr[state_offset] = Scalar[kernel_dtype](
                state_vals[n].cast[kernel_dtype]()
            )

        # Load C values using group_id
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var C_offset = UInt32(
                b * C_strides[0] + group_id * C_strides[1] + n * C_strides[2]
            )
            C_vals[n] = Scalar[kernel_dtype](C.ptr[C_offset]).cast[
                DType.float32
            ]()
        # Compute output
        var out_val = (state_vals * C_vals).reduce_add()

        # Add skip connection
        if has_D:
            var D_offset = UInt32(d * D_strides[0])
            var D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[
                DType.float32
            ]()
            out_val += x_val * D_val

        # Apply gating
        if has_z:
            var z_offset = UInt32(b * z_strides[0] + d * z_strides[1])
            var z_val = Scalar[kernel_dtype](z.ptr[z_offset]).cast[
                DType.float32
            ]()
            out_val *= z_val * sigmoid(z_val)

        # Store output
        var out_offset = UInt32(b * output_strides[0] + d * output_strides[1])
        output.ptr[out_offset] = Scalar[kernel_dtype](
            out_val.cast[kernel_dtype]()
        )

    sync_parallelize[worker](batch * dim)


fn selective_scan_fwd_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    output_strides: Strides3D,
    x_strides: Strides4D,
    out_z_strides: Strides3D,
    u_strides: Strides3D,
    delta_strides: Strides3D,
    A_strides: Strides2D,
    B_strides: Strides4D,
    C_strides: Strides4D,
    D_strides: Strides1D,
    z_strides: Strides3D,
    delta_bias_strides: Strides1D,
):
    """CPU kernel for selective scan forward pass."""

    @parameter
    fn worker(idx: Int):
        var b = idx // dim
        var d = idx % dim

        # Bounds checking
        if b >= batch or d >= dim:
            return

        var group_id = d // group_size

        # Local state storage (max dstate 16)
        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
        var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

        # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var has_delta_bias = delta_bias.dim(0) > 0
        var delta_bias_val = Float32(0.0)
        if has_delta_bias:
            var bias_offset = UInt32(d * delta_bias_strides[0])
            delta_bias_val = Scalar[kernel_dtype](
                delta_bias.ptr[bias_offset]
            ).cast[DType.float32]()

        var has_D = D.dim(0) > 0
        var D_val = Float32(0.0)
        if has_D:
            var D_offset = UInt32(d * D_strides[0])
            D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

        var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
        var has_z = z.dim(0) > 0
        var has_out_z = out_z.dim(0) > 0

        comptime for n in range(DSTATE):
            var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
            A_vals[n] = (
                Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]()
                * LOG2E
            )

        var chunk_size = 2048
        var t_in_chunk = 0
        var chunk_idx = 0

        var curr_u_offset = UInt32(b * u_strides[0] + d * u_strides[1])
        var curr_delta_offset = UInt32(
            b * delta_strides[0] + d * delta_strides[1]
        )
        var curr_output_offset = UInt32(
            b * output_strides[0] + d * output_strides[1]
        )
        var curr_B_offset = UInt32(b * B_strides[0] + group_id * B_strides[1])
        var curr_C_offset = UInt32(b * C_strides[0] + group_id * C_strides[1])
        var curr_z_offset = UInt32(b * z_strides[0] + d * z_strides[1])
        var curr_out_z_offset = UInt32(
            b * out_z_strides[0] + d * out_z_strides[1]
        )
        comptime TILE_SIZE = 4
        var aligned_seqlen = seqlen - (seqlen % TILE_SIZE)
        var t = 0

        while t < aligned_seqlen:
            var u_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var delta_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var z_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)

            if u_strides[2] == 1:
                u_vec = u.ptr.load[width=TILE_SIZE](curr_u_offset)
            else:
                for i in range(TILE_SIZE):
                    u_vec[i] = u.ptr[curr_u_offset + UInt32(i * u_strides[2])]

            if delta_strides[2] == 1:
                delta_vec = delta.ptr.load[width=TILE_SIZE](curr_delta_offset)
            else:
                for i in range(TILE_SIZE):
                    delta_vec[i] = delta.ptr[
                        curr_delta_offset + UInt32(i * delta_strides[2])
                    ]

            if has_z:
                if z_strides[2] == 1:
                    z_vec = z.ptr.load[width=TILE_SIZE](curr_z_offset)
                else:
                    for i in range(TILE_SIZE):
                        z_vec[i] = z.ptr[
                            curr_z_offset + UInt32(i * z_strides[2])
                        ]

            for i in range(TILE_SIZE):
                t_in_chunk += 1

                var u_val = u_vec[i].cast[DType.float32]()
                var delta_val = delta_vec[i].cast[DType.float32]()

                if has_delta_bias:
                    delta_val += delta_bias_val

                if delta_softplus_bool:
                    delta_val = softplus(delta_val)

                var delta_u = delta_val * u_val

                var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

                comptime for n in range(DSTATE):
                    var b_off = (
                        curr_B_offset
                        + UInt32(i * B_strides[3])
                        + UInt32(n * B_strides[2])
                    )
                    var c_off = (
                        curr_C_offset
                        + UInt32(i * C_strides[3])
                        + UInt32(n * C_strides[2])
                    )

                    B_vals[n] = Scalar[kernel_dtype](B.ptr[b_off]).cast[
                        DType.float32
                    ]()
                    C_vals[n] = Scalar[kernel_dtype](C.ptr[c_off]).cast[
                        DType.float32
                    ]()

                var a_t = exp2(A_vals * delta_val)
                var b_t = B_vals * delta_u
                state = state * a_t + b_t
                var output_val = (state * C_vals).reduce_add()

                cum_b = cum_b * a_t + b_t
                cum_a = cum_a * a_t

                if has_D:
                    output_val += D_val * u_val

                var final_val = output_val.cast[kernel_dtype]()

                if has_z:
                    var z_val = z_vec[i].cast[DType.float32]()
                    var out_z_val = output_val * silu(z_val)
                    if has_out_z:
                        var out_z_off = curr_out_z_offset + UInt32(
                            i * out_z_strides[2]
                        )
                        out_z.ptr[out_z_off] = Scalar[kernel_dtype](
                            out_z_val.cast[kernel_dtype]()
                        )
                var out_off = curr_output_offset + UInt32(i * output_strides[2])
                output.ptr[out_off] = Scalar[kernel_dtype](final_val)
                var is_chunk_boundary = t_in_chunk == chunk_size
                var current_t = t + i
                var is_last_step = current_t == seqlen - 1

                if is_chunk_boundary or is_last_step:
                    comptime for n in range(DSTATE):
                        var x_offset_a = UInt32(
                            b * x_strides[0]
                            + d * x_strides[1]
                            + chunk_idx * x_strides[2]
                            + (n * 2) * x_strides[3]
                        )
                        var x_offset_b = UInt32(
                            b * x_strides[0]
                            + d * x_strides[1]
                            + chunk_idx * x_strides[2]
                            + (n * 2 + 1) * x_strides[3]
                        )
                        x.ptr[x_offset_a] = Scalar[kernel_dtype](
                            cum_a[n].cast[kernel_dtype]()
                        )
                        x.ptr[x_offset_b] = Scalar[kernel_dtype](
                            cum_b[n].cast[kernel_dtype]()
                        )
                        cum_a[n] = 1.0
                        cum_b[n] = 0.0

                    if is_chunk_boundary:
                        chunk_idx += 1
                        t_in_chunk = 0

            curr_u_offset += UInt32(u_strides[2] * TILE_SIZE)
            curr_delta_offset += UInt32(delta_strides[2] * TILE_SIZE)
            curr_output_offset += UInt32(output_strides[2] * TILE_SIZE)
            curr_B_offset += UInt32(B_strides[3] * TILE_SIZE)
            curr_C_offset += UInt32(C_strides[3] * TILE_SIZE)
            curr_z_offset += UInt32(z_strides[2] * TILE_SIZE)
            curr_out_z_offset += UInt32(out_z_strides[2] * TILE_SIZE)

            t += TILE_SIZE

        while t < seqlen:
            t_in_chunk += 1
            var u_val = Scalar[kernel_dtype](u.ptr[curr_u_offset]).cast[
                DType.float32
            ]()
            var delta_val = Scalar[kernel_dtype](
                delta.ptr[curr_delta_offset]
            ).cast[DType.float32]()
            if has_delta_bias:
                delta_val += delta_bias_val
            if delta_softplus_bool:
                delta_val = softplus(delta_val)
            var delta_u = delta_val * u_val
            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.ptr[curr_B_offset + UInt32(n * B_strides[2])]
                ).cast[DType.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.ptr[curr_C_offset + UInt32(n * C_strides[2])]
                ).cast[DType.float32]()
            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var output_val = (state * C_vals).reduce_add()
            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t
            if has_D:
                output_val += D_val * u_val
            output.ptr[curr_output_offset] = Scalar[kernel_dtype](
                output_val.cast[kernel_dtype]()
            )
            if has_z:
                var z_val = Scalar[kernel_dtype](z.ptr[curr_z_offset]).cast[
                    DType.float32
                ]()
                var out_z_val = output_val * silu(z_val)
                if has_out_z:
                    out_z.ptr[curr_out_z_offset] = Scalar[kernel_dtype](
                        out_z_val.cast[kernel_dtype]()
                    )

            curr_u_offset += UInt32(u_strides[2])
            curr_delta_offset += UInt32(delta_strides[2])
            curr_output_offset += UInt32(output_strides[2])
            curr_B_offset += UInt32(B_strides[3])
            curr_C_offset += UInt32(C_strides[3])
            curr_z_offset += UInt32(z_strides[2])
            curr_out_z_offset += UInt32(out_z_strides[2])

            var is_chunk_boundary = t_in_chunk == chunk_size
            var is_last_step = t == seqlen - 1
            if is_chunk_boundary or is_last_step:
                comptime for n in range(DSTATE):
                    var x_offset_a = UInt32(
                        b * x_strides[0]
                        + d * x_strides[1]
                        + chunk_idx * x_strides[2]
                        + (n * 2) * x_strides[3]
                    )
                    var x_offset_b = UInt32(
                        b * x_strides[0]
                        + d * x_strides[1]
                        + chunk_idx * x_strides[2]
                        + (n * 2 + 1) * x_strides[3]
                    )
                    x.ptr[x_offset_a] = Scalar[kernel_dtype](
                        cum_a[n].cast[kernel_dtype]()
                    )
                    x.ptr[x_offset_b] = Scalar[kernel_dtype](
                        cum_b[n].cast[kernel_dtype]()
                    )
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0
                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0
            t += 1

    sync_parallelize[worker](batch * dim)


fn selective_scan_fwd_cpu_minimal[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    x_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    output_strides: Strides3D,
    x_strides: Strides4D,
    u_strides: Strides3D,
    delta_strides: Strides3D,
    A_strides: Strides2D,
    B_strides: Strides4D,
    C_strides: Strides4D,
):
    """Minimal CPU kernel for selective scan forward - no D, z, or delta_bias.
    """

    @parameter
    fn worker(idx: Int):
        var b = idx // dim
        var d = idx % dim

        if b >= batch or d >= dim:
            return

        var group_id = d // group_size

        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
        var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var delta_softplus_bool = Bool(Int(delta_softplus) != 0)

        comptime for n in range(DSTATE):
            var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
            A_vals[n] = (
                Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]()
                * LOG2E
            )

        var chunk_size = 2048
        var t_in_chunk = 0
        var chunk_idx = 0

        var curr_u_offset = UInt32(b * u_strides[0] + d * u_strides[1])
        var curr_delta_offset = UInt32(
            b * delta_strides[0] + d * delta_strides[1]
        )
        var curr_output_offset = UInt32(
            b * output_strides[0] + d * output_strides[1]
        )
        var curr_B_offset = UInt32(b * B_strides[0] + group_id * B_strides[1])
        var curr_C_offset = UInt32(b * C_strides[0] + group_id * C_strides[1])

        for t in range(seqlen):
            t_in_chunk += 1

            var u_val = Scalar[kernel_dtype](u.ptr[curr_u_offset]).cast[
                DType.float32
            ]()
            var delta_val = Scalar[kernel_dtype](
                delta.ptr[curr_delta_offset]
            ).cast[DType.float32]()
            if delta_softplus_bool:
                delta_val = softplus(delta_val)
            var delta_u = delta_val * u_val

            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.ptr[curr_B_offset + UInt32(n * B_strides[2])]
                ).cast[DType.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.ptr[curr_C_offset + UInt32(n * C_strides[2])]
                ).cast[DType.float32]()

            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var output_val = (state * C_vals).reduce_add()
            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t
            # No D skip connection
            output.ptr[curr_output_offset] = Scalar[kernel_dtype](
                output_val.cast[kernel_dtype]()
            )
            # No z gating

            curr_u_offset += UInt32(u_strides[2])
            curr_delta_offset += UInt32(delta_strides[2])
            curr_output_offset += UInt32(output_strides[2])
            curr_B_offset += UInt32(B_strides[3])
            curr_C_offset += UInt32(C_strides[3])

            var is_chunk_boundary = t_in_chunk == chunk_size
            var is_last_step = t == seqlen - 1
            if is_chunk_boundary or is_last_step:
                comptime for n in range(DSTATE):
                    var x_offset_a = UInt32(
                        b * x_strides[0]
                        + d * x_strides[1]
                        + chunk_idx * x_strides[2]
                        + (n * 2) * x_strides[3]
                    )
                    var x_offset_b = UInt32(
                        b * x_strides[0]
                        + d * x_strides[1]
                        + chunk_idx * x_strides[2]
                        + (n * 2 + 1) * x_strides[3]
                    )
                    x.ptr[x_offset_a] = Scalar[kernel_dtype](
                        cum_a[n].cast[kernel_dtype]()
                    )
                    x.ptr[x_offset_b] = Scalar[kernel_dtype](
                        cum_b[n].cast[kernel_dtype]()
                    )
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0
                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0

    sync_parallelize[worker](batch * dim)


# ===----------------------------------------------------------------------=== #
# SSD Combined: Selective Scan Discrete Combined Operation
# ===----------------------------------------------------------------------=== #
# This kernel combines selective scan with normalization and residual connections.
# It performs: norm(residual + selective_scan(input))
# This is a fused operation for better performance in Mamba blocks.
# ===----------------------------------------------------------------------=== #


fn ssd_combined_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    residual_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
    gamma_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    residual: LayoutTensor[kernel_dtype, residual_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    gamma: LayoutTensor[kernel_dtype, gamma_layout, MutAnyOrigin],
    epsilon: Scalar[kernel_dtype],
    weight_offset: Scalar[kernel_dtype],
):
    """GPU kernel for SSD combined operation.

    Combines selective scan with normalization and residual connection.
    Performs: norm(residual + selective_scan(input))
    """
    # Compute row-major strides from dimensions
    var n_groups = dim // group_size
    var n_chunks = ceildiv(seqlen, 2048)
    # 3D (batch, dim, seqlen) strides
    var output_b_stride = UInt32(dim * seqlen)
    var output_d_stride = UInt32(seqlen)
    var output_t_stride = UInt32(1)
    var u_b_stride = output_b_stride
    var u_d_stride = output_d_stride
    var u_t_stride = output_t_stride
    var delta_b_stride = output_b_stride
    var delta_d_stride = output_d_stride
    var delta_t_stride = output_t_stride
    var out_z_b_stride = output_b_stride
    var out_z_d_stride = output_d_stride
    var out_z_t_stride = output_t_stride
    var residual_b_stride = output_b_stride
    var residual_d_stride = output_d_stride
    var residual_t_stride = output_t_stride
    var z_b_stride = output_b_stride
    var z_d_stride = output_d_stride
    var z_t_stride = output_t_stride
    # 4D (batch, dim, n_chunks, 2*dstate) strides for x
    var x_b_stride = UInt32(dim * n_chunks * 2 * DSTATE)
    var x_d_stride = UInt32(n_chunks * 2 * DSTATE)
    var x_chunk_stride = UInt32(2 * DSTATE)
    var x_n_stride = UInt32(1)
    # 2D (dim, dstate) strides for A
    var A_d_stride = UInt32(DSTATE)
    var A_n_stride = UInt32(1)
    # 4D (batch, n_groups, dstate, seqlen) strides for B, C
    var B_b_stride = UInt32(n_groups * DSTATE * seqlen)
    var B_g_stride = UInt32(DSTATE * seqlen)
    var B_n_stride = UInt32(seqlen)
    var B_t_stride = UInt32(1)
    var C_b_stride = B_b_stride
    var C_g_stride = B_g_stride
    var C_n_stride = B_n_stride
    var C_t_stride = B_t_stride
    # 1D strides
    var D_stride = UInt32(1)
    var delta_bias_stride = UInt32(1)
    var gamma_stride = UInt32(1)

    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return

    var b = thread_id_int // dim
    var d = thread_id_int % dim

    if b >= batch or d >= dim:
        return

    var group_id = d // group_size

    # Local state storage
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
    var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

    # Pre-load A values
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
    var has_delta_bias = delta_bias.dim(0) > 0
    var delta_bias_val = Float32(0.0)
    if has_delta_bias:
        var bias_offset = UInt32(d) * delta_bias_stride
        delta_bias_val = Scalar[kernel_dtype](delta_bias.ptr[bias_offset]).cast[
            DType.float32
        ]()

    var has_D = D.dim(0) > 0
    var D_val = Float32(0.0)
    if has_D:
        var D_offset = UInt32(d) * D_stride
        D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    var has_z = z.dim(0) > 0
    var has_out_z = out_z.dim(0) > 0

    # Pre-multiply A by LOG2E
    comptime for n in range(DSTATE):
        var A_offset = UInt32(d) * A_d_stride + UInt32(n) * A_n_stride
        A_vals[n] = (
            Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]() * LOG2E
        )

    # Load gamma value for normalization
    var gamma_offset = UInt32(d) * gamma_stride
    var gamma_val = Scalar[kernel_dtype](gamma.ptr[gamma_offset]).cast[
        DType.float32
    ]()
    var epsilon_val = epsilon.cast[DType.float32]()
    var weight_offset_val = weight_offset.cast[DType.float32]()

    var chunk_size = 2048
    var t_in_chunk = 0
    var chunk_idx = 0

    # Initialize running offsets
    var curr_u_offset = UInt32(b) * u_b_stride + UInt32(d) * u_d_stride
    var curr_delta_offset = (
        UInt32(b) * delta_b_stride + UInt32(d) * delta_d_stride
    )
    var curr_output_offset = (
        UInt32(b) * output_b_stride + UInt32(d) * output_d_stride
    )
    var curr_B_offset = UInt32(b) * B_b_stride + UInt32(group_id) * B_g_stride
    var curr_C_offset = UInt32(b) * C_b_stride + UInt32(group_id) * C_g_stride
    var curr_z_offset = UInt32(b) * z_b_stride + UInt32(d) * z_d_stride
    var curr_out_z_offset = (
        UInt32(b) * out_z_b_stride + UInt32(d) * out_z_d_stride
    )
    var curr_residual_offset = (
        UInt32(b) * residual_b_stride + UInt32(d) * residual_d_stride
    )

    # Process sequence
    comptime TILE_SIZE = 8
    var aligned_seqlen = seqlen - (seqlen % TILE_SIZE)
    var t = 0

    while t < aligned_seqlen:
        # Load tile of u, delta, z, residual
        var u_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var delta_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var z_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
        var residual_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)

        if u_t_stride == 1:
            u_vec = u.ptr.load[width=TILE_SIZE](curr_u_offset)
        else:
            for i in range(TILE_SIZE):
                u_vec[i] = u.ptr[curr_u_offset + UInt32(i) * u_t_stride]

        if delta_t_stride == 1:
            delta_vec = delta.ptr.load[width=TILE_SIZE](curr_delta_offset)
        else:
            for i in range(TILE_SIZE):
                delta_vec[i] = delta.ptr[
                    curr_delta_offset + UInt32(i) * delta_t_stride
                ]

        if has_z:
            if z_t_stride == 1:
                z_vec = z.ptr.load[width=TILE_SIZE](curr_z_offset)
            else:
                for i in range(TILE_SIZE):
                    z_vec[i] = z.ptr[curr_z_offset + UInt32(i) * z_t_stride]

        if residual_t_stride == 1:
            residual_vec = residual.ptr.load[width=TILE_SIZE](
                curr_residual_offset
            )
        else:
            for i in range(TILE_SIZE):
                residual_vec[i] = residual.ptr[
                    curr_residual_offset + UInt32(i) * residual_t_stride
                ]

        # Process tile
        for i in range(TILE_SIZE):
            t_in_chunk += 1

            var u_val = u_vec[i].cast[DType.float32]()
            var delta_val = delta_vec[i].cast[DType.float32]()
            var residual_val = residual_vec[i].cast[DType.float32]()

            if has_delta_bias:
                delta_val += delta_bias_val

            if delta_softplus_bool:
                delta_val = softplus(delta_val)

            var delta_u = delta_val * u_val

            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

            comptime for n in range(DSTATE):
                var b_off = (
                    curr_B_offset
                    + UInt32(i) * B_t_stride
                    + UInt32(n) * B_n_stride
                )
                var c_off = (
                    curr_C_offset
                    + UInt32(i) * C_t_stride
                    + UInt32(n) * C_n_stride
                )
                B_vals[n] = Scalar[kernel_dtype](B.ptr[b_off]).cast[
                    DType.float32
                ]()
                C_vals[n] = Scalar[kernel_dtype](C.ptr[c_off]).cast[
                    DType.float32
                ]()

            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var ss_output = (state * C_vals).reduce_add()

            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t

            if has_D:
                ss_output += D_val * u_val

            # Combine with residual and apply element-wise scaling (simplified normalization)
            var combined = residual_val + ss_output
            # Apply gamma scaling (element-wise, not full RMS norm for efficiency)
            var normalized = combined * (gamma_val + weight_offset_val)

            # Apply gating if present
            if has_z:
                var z_val = z_vec[i].cast[DType.float32]()
                var out_z_val = normalized * silu(z_val)
                if has_out_z:
                    var out_z_off = (
                        curr_out_z_offset + UInt32(i) * out_z_t_stride
                    )
                    out_z.ptr[out_z_off] = Scalar[kernel_dtype](
                        out_z_val.cast[kernel_dtype]()
                    )
                normalized = out_z_val

            var out_off = curr_output_offset + UInt32(i) * output_t_stride
            output.ptr[out_off] = Scalar[kernel_dtype](
                normalized.cast[kernel_dtype]()
            )

            # Check chunk boundary
            var is_chunk_boundary = t_in_chunk == chunk_size
            var current_t = t + i
            var is_last_step = current_t == seqlen - 1

            if is_chunk_boundary or is_last_step:
                comptime for n in range(DSTATE):
                    var x_offset_a = UInt32(
                        b * Int(x_b_stride)
                        + d * Int(x_d_stride)
                        + chunk_idx * Int(x_chunk_stride)
                        + (n * 2) * Int(x_n_stride)
                    )
                    var x_offset_b = UInt32(
                        b * Int(x_b_stride)
                        + d * Int(x_d_stride)
                        + chunk_idx * Int(x_chunk_stride)
                        + (n * 2 + 1) * Int(x_n_stride)
                    )
                    x.ptr[x_offset_a] = Scalar[kernel_dtype](
                        cum_a[n].cast[kernel_dtype]()
                    )
                    x.ptr[x_offset_b] = Scalar[kernel_dtype](
                        cum_b[n].cast[kernel_dtype]()
                    )
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0

                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0

        curr_u_offset += u_t_stride * UInt32(TILE_SIZE)
        curr_delta_offset += delta_t_stride * UInt32(TILE_SIZE)
        curr_output_offset += output_t_stride * UInt32(TILE_SIZE)
        curr_B_offset += B_t_stride * UInt32(TILE_SIZE)
        curr_C_offset += C_t_stride * UInt32(TILE_SIZE)
        curr_z_offset += z_t_stride * UInt32(TILE_SIZE)
        curr_out_z_offset += out_z_t_stride * UInt32(TILE_SIZE)
        curr_residual_offset += residual_t_stride * UInt32(TILE_SIZE)

        t += TILE_SIZE

    # Handle remaining timesteps
    while t < seqlen:
        t_in_chunk += 1
        var u_val = Scalar[kernel_dtype](u.ptr[curr_u_offset]).cast[
            DType.float32
        ]()
        var delta_val = Scalar[kernel_dtype](delta.ptr[curr_delta_offset]).cast[
            DType.float32
        ]()
        var residual_val = Scalar[kernel_dtype](
            residual.ptr[curr_residual_offset]
        ).cast[DType.float32]()

        if has_delta_bias:
            delta_val += delta_bias_val
        if delta_softplus_bool:
            delta_val = softplus(delta_val)

        var delta_u = delta_val * u_val
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.ptr[curr_B_offset + UInt32(n) * B_n_stride]
            ).cast[DType.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.ptr[curr_C_offset + UInt32(n) * C_n_stride]
            ).cast[DType.float32]()

        var a_t = exp2(A_vals * delta_val)
        var b_t = B_vals * delta_u
        state = state * a_t + b_t
        var ss_output = (state * C_vals).reduce_add()

        cum_b = cum_b * a_t + b_t
        cum_a = cum_a * a_t

        if has_D:
            ss_output += D_val * u_val

        # Combine with residual and apply element-wise scaling
        var combined = residual_val + ss_output
        var normalized = combined * (gamma_val + weight_offset_val)

        if has_z:
            var z_val = Scalar[kernel_dtype](z.ptr[curr_z_offset]).cast[
                DType.float32
            ]()
            var out_z_val = normalized * silu(z_val)
            if has_out_z:
                out_z.ptr[curr_out_z_offset] = Scalar[kernel_dtype](
                    out_z_val.cast[kernel_dtype]()
                )
            normalized = out_z_val

        output.ptr[curr_output_offset] = Scalar[kernel_dtype](
            normalized.cast[kernel_dtype]()
        )

        curr_u_offset += u_t_stride
        curr_delta_offset += delta_t_stride
        curr_output_offset += output_t_stride
        curr_B_offset += B_t_stride
        curr_C_offset += C_t_stride
        curr_z_offset += z_t_stride
        curr_out_z_offset += out_z_t_stride
        curr_residual_offset += residual_t_stride

        var is_chunk_boundary = t_in_chunk == chunk_size
        var is_last_step = t == seqlen - 1
        if is_chunk_boundary or is_last_step:
            comptime for n in range(DSTATE):
                var x_offset_a = UInt32(
                    b * Int(x_b_stride)
                    + d * Int(x_d_stride)
                    + chunk_idx * Int(x_chunk_stride)
                    + (n * 2) * Int(x_n_stride)
                )
                var x_offset_b = UInt32(
                    b * Int(x_b_stride)
                    + d * Int(x_d_stride)
                    + chunk_idx * Int(x_chunk_stride)
                    + (n * 2 + 1) * Int(x_n_stride)
                )
                x.ptr[x_offset_a] = Scalar[kernel_dtype](
                    cum_a[n].cast[kernel_dtype]()
                )
                x.ptr[x_offset_b] = Scalar[kernel_dtype](
                    cum_b[n].cast[kernel_dtype]()
                )
                cum_a[n] = 1.0
                cum_b[n] = 0.0
            if is_chunk_boundary:
                chunk_idx += 1
                t_in_chunk = 0
        t += 1


fn ssd_combined_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    residual_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
    gamma_layout: Layout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    delta_softplus: Int8,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    residual: LayoutTensor[kernel_dtype, residual_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    gamma: LayoutTensor[kernel_dtype, gamma_layout, MutAnyOrigin],
    epsilon: Scalar[kernel_dtype],
    weight_offset: Scalar[kernel_dtype],
):
    """CPU kernel for SSD combined operation."""
    # Compute row-major strides from dimensions
    var n_groups = dim // group_size
    var n_chunks = ceildiv(seqlen, 2048)
    # 3D (batch, dim, seqlen) strides
    var output_b_stride = UInt32(dim * seqlen)
    var output_d_stride = UInt32(seqlen)
    var output_t_stride = UInt32(1)
    var u_b_stride = output_b_stride
    var u_d_stride = output_d_stride
    var u_t_stride = output_t_stride
    var delta_b_stride = output_b_stride
    var delta_d_stride = output_d_stride
    var delta_t_stride = output_t_stride
    var out_z_b_stride = output_b_stride
    var out_z_d_stride = output_d_stride
    var out_z_t_stride = output_t_stride
    var residual_b_stride = output_b_stride
    var residual_d_stride = output_d_stride
    var residual_t_stride = output_t_stride
    var z_b_stride = output_b_stride
    var z_d_stride = output_d_stride
    var z_t_stride = output_t_stride
    # 4D (batch, dim, n_chunks, 2*dstate) strides for x
    var x_b_stride = UInt32(dim * n_chunks * 2 * DSTATE)
    var x_d_stride = UInt32(n_chunks * 2 * DSTATE)
    var x_chunk_stride = UInt32(2 * DSTATE)
    var x_n_stride = UInt32(1)
    # 2D (dim, dstate) strides for A
    var A_d_stride = UInt32(DSTATE)
    var A_n_stride = UInt32(1)
    # 4D (batch, n_groups, dstate, seqlen) strides for B, C
    var B_b_stride = UInt32(n_groups * DSTATE * seqlen)
    var B_g_stride = UInt32(DSTATE * seqlen)
    var B_n_stride = UInt32(seqlen)
    var B_t_stride = UInt32(1)
    var C_b_stride = B_b_stride
    var C_g_stride = B_g_stride
    var C_n_stride = B_n_stride
    var C_t_stride = B_t_stride
    # 1D strides
    var D_stride = UInt32(1)
    var delta_bias_stride = UInt32(1)
    var gamma_stride = UInt32(1)

    @parameter
    fn worker(idx: Int):
        var b = idx // dim
        var d = idx % dim

        var group_id = d // group_size

        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
        var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var has_delta_bias = delta_bias.dim(0) > 0
        var delta_bias_val = Float32(0.0)
        if has_delta_bias:
            var bias_offset = UInt32(d) * delta_bias_stride
            delta_bias_val = Scalar[kernel_dtype](
                delta_bias.ptr[bias_offset]
            ).cast[DType.float32]()

        var has_D = D.dim(0) > 0
        var D_val = Float32(0.0)
        if has_D:
            var D_offset = UInt32(d) * D_stride
            D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

        var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
        var has_z = z.dim(0) > 0
        var has_out_z = out_z.dim(0) > 0

        comptime for n in range(DSTATE):
            var A_offset = UInt32(d) * A_d_stride + UInt32(n) * A_n_stride
            A_vals[n] = (
                Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]()
                * LOG2E
            )

        var gamma_offset = UInt32(d) * gamma_stride
        var gamma_val = Scalar[kernel_dtype](gamma.ptr[gamma_offset]).cast[
            DType.float32
        ]()
        var epsilon_val = epsilon.cast[DType.float32]()
        var weight_offset_val = weight_offset.cast[DType.float32]()

        var chunk_size = 2048
        var t_in_chunk = 0
        var chunk_idx = 0

        var curr_u_offset = UInt32(b) * u_b_stride + UInt32(d) * u_d_stride
        var curr_delta_offset = (
            UInt32(b) * delta_b_stride + UInt32(d) * delta_d_stride
        )
        var curr_output_offset = (
            UInt32(b) * output_b_stride + UInt32(d) * output_d_stride
        )
        var curr_B_offset = (
            UInt32(b) * B_b_stride + UInt32(group_id) * B_g_stride
        )
        var curr_C_offset = (
            UInt32(b) * C_b_stride + UInt32(group_id) * C_g_stride
        )
        var curr_z_offset = UInt32(b) * z_b_stride + UInt32(d) * z_d_stride
        var curr_out_z_offset = (
            UInt32(b) * out_z_b_stride + UInt32(d) * out_z_d_stride
        )
        var curr_residual_offset = (
            UInt32(b) * residual_b_stride + UInt32(d) * residual_d_stride
        )
        comptime TILE_SIZE = 4
        var aligned_seqlen = seqlen - (seqlen % TILE_SIZE)
        var t = 0

        while t < aligned_seqlen:
            var u_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var delta_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var z_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)
            var residual_vec = SIMD[kernel_dtype, TILE_SIZE](0.0)

            if u_t_stride == 1:
                u_vec = u.ptr.load[width=TILE_SIZE](curr_u_offset)
            else:
                for i in range(TILE_SIZE):
                    u_vec[i] = u.ptr[curr_u_offset + UInt32(i) * u_t_stride]

            if delta_t_stride == 1:
                delta_vec = delta.ptr.load[width=TILE_SIZE](curr_delta_offset)
            else:
                for i in range(TILE_SIZE):
                    delta_vec[i] = delta.ptr[
                        curr_delta_offset + UInt32(i) * delta_t_stride
                    ]

            if has_z:
                if z_t_stride == 1:
                    z_vec = z.ptr.load[width=TILE_SIZE](curr_z_offset)
                else:
                    for i in range(TILE_SIZE):
                        z_vec[i] = z.ptr[curr_z_offset + UInt32(i) * z_t_stride]

            if residual_t_stride == 1:
                residual_vec = residual.ptr.load[width=TILE_SIZE](
                    curr_residual_offset
                )
            else:
                for i in range(TILE_SIZE):
                    residual_vec[i] = residual.ptr[
                        curr_residual_offset + UInt32(i) * residual_t_stride
                    ]

            for i in range(TILE_SIZE):
                t_in_chunk += 1

                var u_val = u_vec[i].cast[DType.float32]()
                var delta_val = delta_vec[i].cast[DType.float32]()
                var residual_val = residual_vec[i].cast[DType.float32]()

                if has_delta_bias:
                    delta_val += delta_bias_val

                if delta_softplus_bool:
                    delta_val = softplus(delta_val)

                var delta_u = delta_val * u_val

                var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

                comptime for n in range(DSTATE):
                    var b_off = (
                        curr_B_offset
                        + UInt32(i) * B_t_stride
                        + UInt32(n) * B_n_stride
                    )
                    var c_off = (
                        curr_C_offset
                        + UInt32(i) * C_t_stride
                        + UInt32(n) * C_n_stride
                    )
                    B_vals[n] = Scalar[kernel_dtype](B.ptr[b_off]).cast[
                        DType.float32
                    ]()
                    C_vals[n] = Scalar[kernel_dtype](C.ptr[c_off]).cast[
                        DType.float32
                    ]()

                var a_t = exp2(A_vals * delta_val)
                var b_t = B_vals * delta_u
                state = state * a_t + b_t
                var ss_output = (state * C_vals).reduce_add()

                cum_b = cum_b * a_t + b_t
                cum_a = cum_a * a_t

                if has_D:
                    ss_output += D_val * u_val

                # Combine with residual and apply element-wise scaling
                var combined = residual_val + ss_output
                var normalized = combined * (gamma_val + weight_offset_val)

                if has_z:
                    var z_val = z_vec[i].cast[DType.float32]()
                    var out_z_val = normalized * silu(z_val)
                    if has_out_z:
                        var out_z_off = (
                            curr_out_z_offset + UInt32(i) * out_z_t_stride
                        )
                        out_z.ptr[out_z_off] = Scalar[kernel_dtype](
                            out_z_val.cast[kernel_dtype]()
                        )
                    normalized = out_z_val

                var out_off = curr_output_offset + UInt32(i) * output_t_stride
                output.ptr[out_off] = Scalar[kernel_dtype](
                    normalized.cast[kernel_dtype]()
                )
                var is_chunk_boundary = t_in_chunk == chunk_size
                var current_t = t + i
                var is_last_step = current_t == seqlen - 1

                if is_chunk_boundary or is_last_step:
                    comptime for n in range(DSTATE):
                        var x_offset_a = UInt32(
                            b * Int(x_b_stride)
                            + d * Int(x_d_stride)
                            + chunk_idx * Int(x_chunk_stride)
                            + (n * 2) * Int(x_n_stride)
                        )
                        var x_offset_b = UInt32(
                            b * Int(x_b_stride)
                            + d * Int(x_d_stride)
                            + chunk_idx * Int(x_chunk_stride)
                            + (n * 2 + 1) * Int(x_n_stride)
                        )
                        x.ptr[x_offset_a] = Scalar[kernel_dtype](
                            cum_a[n].cast[kernel_dtype]()
                        )
                        x.ptr[x_offset_b] = Scalar[kernel_dtype](
                            cum_b[n].cast[kernel_dtype]()
                        )
                        cum_a[n] = 1.0
                        cum_b[n] = 0.0

                    if is_chunk_boundary:
                        chunk_idx += 1
                        t_in_chunk = 0

            curr_u_offset += u_t_stride * UInt32(TILE_SIZE)
            curr_delta_offset += delta_t_stride * UInt32(TILE_SIZE)
            curr_output_offset += output_t_stride * UInt32(TILE_SIZE)
            curr_B_offset += B_t_stride * UInt32(TILE_SIZE)
            curr_C_offset += C_t_stride * UInt32(TILE_SIZE)
            curr_z_offset += z_t_stride * UInt32(TILE_SIZE)
            curr_out_z_offset += out_z_t_stride * UInt32(TILE_SIZE)
            curr_residual_offset += residual_t_stride * UInt32(TILE_SIZE)

            t += TILE_SIZE

        while t < seqlen:
            t_in_chunk += 1
            var u_val = Scalar[kernel_dtype](u.ptr[curr_u_offset]).cast[
                DType.float32
            ]()
            var delta_val = Scalar[kernel_dtype](
                delta.ptr[curr_delta_offset]
            ).cast[DType.float32]()
            var residual_val = Scalar[kernel_dtype](
                residual.ptr[curr_residual_offset]
            ).cast[DType.float32]()

            if has_delta_bias:
                delta_val += delta_bias_val
            if delta_softplus_bool:
                delta_val = softplus(delta_val)

            var delta_u = delta_val * u_val
            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.ptr[curr_B_offset + UInt32(n) * B_n_stride]
                ).cast[DType.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.ptr[curr_C_offset + UInt32(n) * C_n_stride]
                ).cast[DType.float32]()

            var a_t = exp2(A_vals * delta_val)
            var b_t = B_vals * delta_u
            state = state * a_t + b_t
            var ss_output = (state * C_vals).reduce_add()

            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t

            if has_D:
                ss_output += D_val * u_val

            # Combine with residual and apply element-wise scaling
            var combined = residual_val + ss_output
            var normalized = combined * (gamma_val + weight_offset_val)

            if has_z:
                var z_val = Scalar[kernel_dtype](z.ptr[curr_z_offset]).cast[
                    DType.float32
                ]()
                var out_z_val = normalized * silu(z_val)
                if has_out_z:
                    out_z.ptr[curr_out_z_offset] = Scalar[kernel_dtype](
                        out_z_val.cast[kernel_dtype]()
                    )
                normalized = out_z_val

            output.ptr[curr_output_offset] = Scalar[kernel_dtype](
                normalized.cast[kernel_dtype]()
            )
            curr_u_offset += u_t_stride
            curr_delta_offset += delta_t_stride
            curr_output_offset += output_t_stride
            curr_B_offset += B_t_stride
            curr_C_offset += C_t_stride
            curr_z_offset += z_t_stride
            curr_out_z_offset += out_z_t_stride
            curr_residual_offset += residual_t_stride

            var is_chunk_boundary = t_in_chunk == chunk_size
            var is_last_step = t == seqlen - 1
            if is_chunk_boundary or is_last_step:
                comptime for n in range(DSTATE):
                    var x_offset_a = UInt32(
                        b * Int(x_b_stride)
                        + d * Int(x_d_stride)
                        + chunk_idx * Int(x_chunk_stride)
                        + (n * 2) * Int(x_n_stride)
                    )
                    var x_offset_b = UInt32(
                        b * Int(x_b_stride)
                        + d * Int(x_d_stride)
                        + chunk_idx * Int(x_chunk_stride)
                        + (n * 2 + 1) * Int(x_n_stride)
                    )
                    x.ptr[x_offset_a] = Scalar[kernel_dtype](
                        cum_a[n].cast[kernel_dtype]()
                    )
                    x.ptr[x_offset_b] = Scalar[kernel_dtype](
                        cum_b[n].cast[kernel_dtype]()
                    )
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0
                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0
            t += 1

    sync_parallelize[worker](batch * dim)


# ===----------------------------------------------------------------------=== #
# Mamba Split Conv1D Scan Combined: Fused operation combining conv1d, split, and scan
# ===----------------------------------------------------------------------=== #
# This kernel performs:
# 1. Split input zxbcdt into z, xBC, dt
# 2. Apply causal conv1d to xBC with activation
# 3. Split conv output into x, B, C
# 4. Apply selective scan
# 5. Optionally apply RMSNorm with gating
# 6. Optionally apply output projection
# ===----------------------------------------------------------------------=== #


fn mamba_split_conv1d_scan_combined_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
    zxbcdt_layout: Layout,
    conv_weight_layout: Layout,
    conv_bias_layout: Layout,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
    rmsnorm_weight_layout: Layout,
    outproj_weight_layout: Layout,
    outproj_bias_layout: Layout,
](
    batch: Int,
    seqlen: Int,
    dim: Int,  # Total dimension = nheads * headdim
    nheads: Int,
    headdim: Int,
    ngroups: Int,
    width: Int,  # Conv kernel width
    chunk_size: Int,
    delta_softplus: Int8,
    norm_before_gate: Int8,
    has_rmsnorm: Int8,
    has_outproj: Int8,
    zxbcdt: LayoutTensor[
        kernel_dtype, zxbcdt_layout, MutAnyOrigin
    ],  # (batch, seqlen, 2*dim + 2*ngroups*dstate + nheads)
    conv_weight: LayoutTensor[
        kernel_dtype, conv_weight_layout, MutAnyOrigin
    ],  # (dim + 2*ngroups*dstate, width)
    conv_bias: LayoutTensor[
        kernel_dtype, conv_bias_layout, MutAnyOrigin
    ],  # (dim + 2*ngroups*dstate,)
    dt_bias: LayoutTensor[
        kernel_dtype, delta_bias_layout, MutAnyOrigin
    ],  # (nheads,)
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],  # (nheads,)
    D: LayoutTensor[
        kernel_dtype, D_layout, MutAnyOrigin
    ],  # (nheads, headdim) or (nheads,)
    x: LayoutTensor[
        kernel_dtype, x_layout, MutAnyOrigin
    ],  # (batch, dim, num_chunks, 2*dstate)
    out_z: LayoutTensor[
        kernel_dtype, out_z_layout, MutAnyOrigin
    ],  # (batch, dim, seqlen)
    dt: LayoutTensor[
        kernel_dtype, dt_layout, MutAnyOrigin
    ],  # (batch, nheads, seqlen)
    B: LayoutTensor[
        kernel_dtype, B_layout, MutAnyOrigin
    ],  # (batch, ngroups, dstate, seqlen)
    C: LayoutTensor[
        kernel_dtype, C_layout, MutAnyOrigin
    ],  # (batch, ngroups, dstate, seqlen)
    z: LayoutTensor[
        kernel_dtype, z_layout, MutAnyOrigin
    ],  # (batch, dim, seqlen)
    rmsnorm_weight: LayoutTensor[
        kernel_dtype, rmsnorm_weight_layout, MutAnyOrigin
    ],  # (dim,)
    outproj_weight: LayoutTensor[
        kernel_dtype, outproj_weight_layout, MutAnyOrigin
    ],  # (out_dim, dim)
    outproj_bias: LayoutTensor[
        kernel_dtype, outproj_bias_layout, MutAnyOrigin
    ],  # (out_dim,)
    output: LayoutTensor[
        kernel_dtype, output_layout, MutAnyOrigin
    ],  # (batch, seqlen, dim) or (batch, seqlen, out_dim)
    epsilon: Scalar[kernel_dtype],
):
    """CPU kernel for mamba_split_conv1d_scan_combined operation.

    Input zxbcdt structure:
    - Channels 0 to dim-1: z (gating values)
    - Channels dim to dim + 2*ngroups*dstate - 1: xBC (x, B, C before conv)
    - Channels 2*dim + 2*ngroups*dstate to end: dt (time step values)

    After conv on xBC:
    - Channels 0 to dim-1: x (input to scan)
    - Channels dim to dim + ngroups*dstate - 1: B
    - Channels dim + ngroups*dstate to dim + 2*ngroups*dstate - 1: C
    """
    # Compute row-major strides from dimensions
    var n_chunks = ceildiv(seqlen, chunk_size)
    var zxbcdt_channels = 2 * dim + 2 * ngroups * DSTATE + nheads
    var out_dim = output.dim(2)
    # zxbcdt: (batch, seqlen, channels)
    var zxbcdt_b_stride = UInt32(seqlen * zxbcdt_channels)
    var zxbcdt_s_stride = UInt32(zxbcdt_channels)
    var zxbcdt_c_stride = UInt32(1)
    # conv_weight: (channels, width)
    var conv_weight_c_stride = UInt32(width)
    var conv_weight_w_stride = UInt32(1)
    # conv_bias: (channels,)
    var conv_bias_stride = UInt32(1)
    # output: (batch, seqlen, out_dim)
    var output_b_stride = UInt32(seqlen * out_dim)
    var output_s_stride = UInt32(out_dim)
    var output_c_stride = UInt32(1)
    # x: (batch, dim, n_chunks, 2*dstate)
    var x_b_stride = UInt32(dim * n_chunks * 2 * DSTATE)
    var x_d_stride = UInt32(n_chunks * 2 * DSTATE)
    var x_chunk_stride = UInt32(2 * DSTATE)
    var x_n_stride = UInt32(1)
    # out_z, z: (batch, dim, seqlen)
    var out_z_b_stride = UInt32(dim * seqlen)
    var out_z_d_stride = UInt32(seqlen)
    var out_z_t_stride = UInt32(1)
    var z_b_stride = out_z_b_stride
    var z_d_stride = out_z_d_stride
    var z_t_stride = out_z_t_stride
    # dt: (batch, nheads, seqlen)
    var dt_b_stride = UInt32(nheads * seqlen)
    var dt_h_stride = UInt32(seqlen)
    var dt_s_stride = UInt32(1)
    # A: (nheads,)
    var A_stride = UInt32(1)
    # B, C: (batch, ngroups, dstate, seqlen)
    var B_b_stride = UInt32(ngroups * DSTATE * seqlen)
    var B_g_stride = UInt32(DSTATE * seqlen)
    var B_n_stride = UInt32(seqlen)
    var B_t_stride = UInt32(1)
    var C_b_stride = B_b_stride
    var C_g_stride = B_g_stride
    var C_n_stride = B_n_stride
    var C_t_stride = B_t_stride
    # D: (nheads, headdim) or (nheads,)
    var D_h_stride = UInt32(headdim) if D.dim(1) > 0 else UInt32(1)
    var D_p_stride = UInt32(1)
    # 1D strides
    var dt_bias_stride = UInt32(1)
    var rmsnorm_weight_stride = UInt32(1)
    # outproj_weight: (out_dim, dim)
    var outproj_weight_out_stride = UInt32(dim)
    var outproj_weight_in_stride = UInt32(1)
    var outproj_bias_stride = UInt32(1)

    var width_minus_1 = width - 1
    var group_size = dim // nheads  # Should equal headdim
    var z_start = 0
    var xBC_start = dim
    var dt_start = 2 * dim + 2 * ngroups * DSTATE

    @parameter
    fn worker(idx: Int) raises:
        var b = idx // dim
        var d = idx % dim
        var h = d // headdim
        var p = d % headdim
        var group_id = h // ngroups if ngroups > 1 else 0

        # Initialize state for selective scan
        var state = SIMD[DType.float32, MAX_DSTATE](0.0)
        var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
        var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

        # Pre-load A values
        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var A_offset = UInt32(h) * A_stride
            A_vals[n] = (
                Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]()
                * LOG2E
            )

        var has_D = D.dim(0) > 0
        var D_val = Float32(0.0)
        if has_D:
            if D.dim(1) > 0:
                # D is (nheads, headdim)
                var D_offset = UInt32(h) * D_h_stride + UInt32(p) * D_p_stride
                D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[
                    DType.float32
                ]()
            else:
                # D is (nheads,)
                var D_offset = UInt32(h) * D_h_stride
                D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[
                    DType.float32
                ]()

        var has_dt_bias = dt_bias.dim(0) > 0
        var dt_bias_val = Float32(0.0)
        if has_dt_bias:
            var bias_offset = UInt32(h) * dt_bias_stride
            dt_bias_val = Scalar[kernel_dtype](dt_bias.ptr[bias_offset]).cast[
                DType.float32
            ]()

        var chunk_idx = 0
        var t_in_chunk = 0

        # Process sequence
        for t in range(seqlen):
            # Step 1: Load z and dt from zxbcdt
            var z_channel = z_start + d
            var z_offset = (
                UInt32(b) * zxbcdt_b_stride
                + UInt32(t) * zxbcdt_s_stride
                + UInt32(z_channel) * zxbcdt_c_stride
            )
            var z_val = Scalar[kernel_dtype](zxbcdt.ptr[z_offset]).cast[
                DType.float32
            ]()

            var dt_channel = dt_start + h
            var dt_offset = (
                UInt32(b) * zxbcdt_b_stride
                + UInt32(t) * zxbcdt_s_stride
                + UInt32(dt_channel) * zxbcdt_c_stride
            )
            var dt_val = Scalar[kernel_dtype](zxbcdt.ptr[dt_offset]).cast[
                DType.float32
            ]()
            dt_val = dt_val + dt_bias_val
            if Bool(Int(delta_softplus) != 0):
                dt_val = softplus(dt_val)

            # Store dt
            var dt_out_offset = (
                UInt32(b) * dt_b_stride
                + UInt32(h) * dt_h_stride
                + UInt32(t) * dt_s_stride
            )
            dt.ptr[dt_out_offset] = Scalar[kernel_dtype](
                dt_val.cast[kernel_dtype]()
            )

            # Step 2: Compute conv for x channel (d is the x channel index)
            var x_channel_in_xBC = d  # x channel in xBC space (0 to dim-1)
            var xBC_channel_in_zxbcdt = xBC_start + x_channel_in_xBC

            var conv_sum = Scalar[kernel_dtype](
                conv_bias.ptr[UInt32(x_channel_in_xBC) * conv_bias_stride]
            ).cast[DType.float32]()

            for w in range(width):
                var input_t = t - (width_minus_1 - w)
                if input_t >= 0:
                    var xbc_offset = (
                        UInt32(b) * zxbcdt_b_stride
                        + UInt32(input_t) * zxbcdt_s_stride
                        + UInt32(xBC_channel_in_zxbcdt) * zxbcdt_c_stride
                    )
                    var input_val = Scalar[kernel_dtype](
                        zxbcdt.ptr[xbc_offset]
                    ).cast[DType.float32]()
                    var weight_offset = (
                        UInt32(x_channel_in_xBC) * conv_weight_c_stride
                        + UInt32(w) * conv_weight_w_stride
                    )
                    var weight_val = Scalar[kernel_dtype](
                        conv_weight.ptr[weight_offset]
                    ).cast[DType.float32]()
                    conv_sum = conv_sum + input_val * weight_val

            # Apply SiLU activation
            var x_val = conv_sum / (1.0 + exp(-conv_sum))

            # Step 3: Compute B and C for this group
            var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

            comptime for n in range(DSTATE):
                # B channel: dim + group_id * dstate + n
                var B_channel_in_xBC = dim + group_id * DSTATE + n
                var B_channel_in_zxbcdt = xBC_start + B_channel_in_xBC

                var B_conv_sum = Scalar[kernel_dtype](
                    conv_bias.ptr[UInt32(B_channel_in_xBC) * conv_bias_stride]
                ).cast[DType.float32]()
                for w in range(width):
                    var input_t = t - (width_minus_1 - w)
                    if input_t >= 0:
                        var xbc_offset = (
                            UInt32(b) * zxbcdt_b_stride
                            + UInt32(input_t) * zxbcdt_s_stride
                            + UInt32(B_channel_in_zxbcdt) * zxbcdt_c_stride
                        )
                        var input_val = Scalar[kernel_dtype](
                            zxbcdt.ptr[xbc_offset]
                        ).cast[DType.float32]()
                        var weight_offset = (
                            UInt32(B_channel_in_xBC) * conv_weight_c_stride
                            + UInt32(w) * conv_weight_w_stride
                        )
                        var weight_val = Scalar[kernel_dtype](
                            conv_weight.ptr[weight_offset]
                        ).cast[DType.float32]()
                        B_conv_sum = B_conv_sum + input_val * weight_val
                B_vals[n] = B_conv_sum / (1.0 + math.exp(-B_conv_sum))  # SiLU

                # Store B
                var B_offset = (
                    UInt32(b) * B_b_stride
                    + UInt32(group_id) * B_g_stride
                    + UInt32(n) * B_n_stride
                    + UInt32(t) * B_t_stride
                )
                B.ptr[B_offset] = Scalar[kernel_dtype](
                    B_vals[n].cast[kernel_dtype]()
                )
                # C channel: dim + ngroups*dstate + group_id * dstate + n
                var C_channel_in_xBC = (
                    dim + ngroups * DSTATE + group_id * DSTATE + n
                )
                var C_channel_in_zxbcdt = xBC_start + C_channel_in_xBC

                var C_conv_sum = Scalar[kernel_dtype](
                    conv_bias.ptr[UInt32(C_channel_in_xBC) * conv_bias_stride]
                ).cast[DType.float32]()
                for w in range(width):
                    var input_t = t - (width_minus_1 - w)
                    if input_t >= 0:
                        var xbc_offset = (
                            UInt32(b) * zxbcdt_b_stride
                            + UInt32(input_t) * zxbcdt_s_stride
                            + UInt32(C_channel_in_zxbcdt) * zxbcdt_c_stride
                        )
                        var input_val = Scalar[kernel_dtype](
                            zxbcdt.ptr[xbc_offset]
                        ).cast[DType.float32]()
                        var weight_offset = (
                            UInt32(C_channel_in_xBC) * conv_weight_c_stride
                            + UInt32(w) * conv_weight_w_stride
                        )
                        var weight_val = Scalar[kernel_dtype](
                            conv_weight.ptr[weight_offset]
                        ).cast[DType.float32]()
                        C_conv_sum = C_conv_sum + input_val * weight_val
                C_vals[n] = C_conv_sum / (1.0 + math.exp(-C_conv_sum))  # SiLU

                # Store C
                var C_offset = (
                    UInt32(b) * C_b_stride
                    + UInt32(group_id) * C_g_stride
                    + UInt32(n) * C_n_stride
                    + UInt32(t) * C_t_stride
                )
                C.ptr[C_offset] = Scalar[kernel_dtype](
                    C_vals[n].cast[kernel_dtype]()
                )
            # Step 4: Selective scan computation
            var a_t = exp2(dt_val * A_vals)
            var b_t = B_vals * dt_val

            state = state * a_t + b_t
            var ss_output = (state * C_vals).reduce_add()

            cum_b = cum_b * a_t + b_t
            cum_a = cum_a * a_t

            if has_D:
                ss_output += D_val * x_val

            # Step 5: Apply gating and normalization
            var out_val = ss_output
            if has_rmsnorm:
                var rmsnorm_w = Scalar[kernel_dtype](
                    rmsnorm_weight.ptr[UInt32(d) * rmsnorm_weight_stride]
                ).cast[DType.float32]()
                var epsilon_val = Scalar[kernel_dtype](epsilon).cast[
                    DType.float32
                ]()
                if norm_before_gate:
                    # RMSNorm(x) * SiLU(z)
                    var norm_val = (
                        rsqrt(out_val * out_val + epsilon_val) * rmsnorm_w
                    )
                    out_val = out_val * norm_val * silu(z_val)
                else:
                    # RMSNorm(x * SiLU(z))
                    var gated = out_val * silu(z_val)
                    var norm_val = (
                        rsqrt(gated * gated + epsilon_val) * rmsnorm_w
                    )
                    out_val = gated * norm_val
            else:
                # Just gating
                out_val = out_val * silu(z_val)

            # Store z and out_z
            var z_out_offset = (
                UInt32(b) * z_b_stride
                + UInt32(d) * z_d_stride
                + UInt32(t) * z_t_stride
            )
            z.ptr[z_out_offset] = Scalar[kernel_dtype](
                z_val.cast[kernel_dtype]()
            )
            if out_z.dim(0) > 0:
                var out_z_offset = (
                    UInt32(b) * out_z_b_stride
                    + UInt32(d) * out_z_d_stride
                    + UInt32(t) * out_z_t_stride
                )
                out_z.ptr[out_z_offset] = Scalar[kernel_dtype](
                    out_val.cast[kernel_dtype]()
                )
            # Step 6: Output projection (if present)
            if has_outproj:
                # Output projection: output[b, t, o] = sum(input[b, t, d] * weight[o, d] for all d) + bias[o]
                # Note: This implementation processes one d at a time and accumulates contributions.
                # For proper thread safety, output should be initialized to bias before this kernel runs,
                # or use a different parallelization strategy (e.g., process (b, t, o) instead of (b, d)).
                var out_dim = output.dim(2)
                for o in range(out_dim):
                    # Load weight[o, d]
                    var weight_offset = (
                        UInt32(o) * outproj_weight_out_stride
                        + UInt32(d) * outproj_weight_in_stride
                    )
                    var weight_val = Scalar[kernel_dtype](
                        outproj_weight.ptr[weight_offset]
                    ).cast[DType.float32]()

                    # Compute contribution: input[b, t, d] * weight[o, d]
                    var contribution = out_val * weight_val

                    # Get output location
                    var out_o_offset = (
                        UInt32(b) * output_b_stride
                        + UInt32(t) * output_s_stride
                        + UInt32(o) * output_c_stride
                    )
                    # Initialize with bias on first d, then accumulate contributions
                    if d == 0:
                        var bias_val = Float32(0.0)
                        if outproj_bias.dim(0) > 0:
                            var bias_offset = UInt32(o) * outproj_bias_stride
                            bias_val = Scalar[kernel_dtype](
                                outproj_bias.ptr[bias_offset]
                            ).cast[DType.float32]()
                        output.ptr[out_o_offset] = Scalar[kernel_dtype](
                            (bias_val + contribution).cast[kernel_dtype]()
                        )
                    else:
                        # Read-modify-write: load current value, add contribution, store
                        # Note: This has a race condition when multiple threads write to same location.
                        # For correctness, output should be pre-initialized or use atomic operations.
                        var current_out = Scalar[kernel_dtype](
                            output.ptr[out_o_offset]
                        ).cast[DType.float32]()
                        current_out = current_out + contribution
                        output.ptr[out_o_offset] = Scalar[kernel_dtype](
                            current_out.cast[kernel_dtype]()
                        )
            else:
                # No output projection - store directly
                var out_offset = (
                    UInt32(b) * output_b_stride
                    + UInt32(t) * output_s_stride
                    + UInt32(d) * output_c_stride
                )
                output.ptr[out_offset] = Scalar[kernel_dtype](
                    out_val.cast[kernel_dtype]()
                )
            # Check chunk boundary
            t_in_chunk += 1
            var is_chunk_boundary = t_in_chunk == chunk_size
            var is_last_step = t == seqlen - 1

            if is_chunk_boundary or is_last_step:
                comptime for n in range(DSTATE):
                    var x_offset_a = UInt32(
                        b * Int(x_b_stride)
                        + d * Int(x_d_stride)
                        + chunk_idx * Int(x_chunk_stride)
                        + (n * 2) * Int(x_n_stride)
                    )
                    var x_offset_b = UInt32(
                        b * Int(x_b_stride)
                        + d * Int(x_d_stride)
                        + chunk_idx * Int(x_chunk_stride)
                        + (n * 2 + 1) * Int(x_n_stride)
                    )
                    x.ptr[x_offset_a] = Scalar[kernel_dtype](
                        cum_a[n].cast[kernel_dtype]()
                    )
                    x.ptr[x_offset_b] = Scalar[kernel_dtype](
                        cum_b[n].cast[kernel_dtype]()
                    )
                    cum_a[n] = 1.0
                    cum_b[n] = 0.0

                if is_chunk_boundary:
                    chunk_idx += 1
                    t_in_chunk = 0

    sync_parallelize[worker](batch * dim)


fn mamba_split_conv1d_scan_combined_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    zxbcdt_layout: Layout,
    conv_weight_layout: Layout,
    conv_bias_layout: Layout,
    output_layout: Layout,
    x_layout: Layout,
    out_z_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
    rmsnorm_weight_layout: Layout,
    outproj_weight_layout: Layout,
    outproj_bias_layout: Layout,
](
    total_batch_dim: Int,
    batch: Int,
    seqlen: Int,
    dim: Int,
    nheads: Int,
    headdim: Int,
    ngroups: Int,
    width: Int,
    chunk_size: Int,
    delta_softplus: Int8,
    norm_before_gate: Int8,
    has_rmsnorm: Int8,
    has_outproj: Int8,
    zxbcdt: LayoutTensor[kernel_dtype, zxbcdt_layout, MutAnyOrigin],
    conv_weight: LayoutTensor[kernel_dtype, conv_weight_layout, MutAnyOrigin],
    conv_bias: LayoutTensor[kernel_dtype, conv_bias_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    out_z: LayoutTensor[kernel_dtype, out_z_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    rmsnorm_weight: LayoutTensor[
        kernel_dtype, rmsnorm_weight_layout, MutAnyOrigin
    ],
    outproj_weight: LayoutTensor[
        kernel_dtype, outproj_weight_layout, MutAnyOrigin
    ],
    outproj_bias: LayoutTensor[kernel_dtype, outproj_bias_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    epsilon: Scalar[kernel_dtype],
):
    """GPU kernel for mamba_split_conv1d_scan_combined operation."""
    # Compute row-major strides from dimensions
    var n_chunks = ceildiv(seqlen, chunk_size)
    var zxbcdt_channels = 2 * dim + 2 * ngroups * DSTATE + nheads
    var out_dim = output.dim(2)
    # zxbcdt: (batch, seqlen, channels)
    var zxbcdt_b_stride = UInt32(seqlen * zxbcdt_channels)
    var zxbcdt_s_stride = UInt32(zxbcdt_channels)
    var zxbcdt_c_stride = UInt32(1)
    # conv_weight: (channels, width)
    var conv_weight_c_stride = UInt32(width)
    var conv_weight_w_stride = UInt32(1)
    # conv_bias: (channels,)
    var conv_bias_stride = UInt32(1)
    # output: (batch, seqlen, out_dim)
    var output_b_stride = UInt32(seqlen * out_dim)
    var output_s_stride = UInt32(out_dim)
    var output_c_stride = UInt32(1)
    # x: (batch, dim, n_chunks, 2*dstate)
    var x_b_stride = UInt32(dim * n_chunks * 2 * DSTATE)
    var x_d_stride = UInt32(n_chunks * 2 * DSTATE)
    var x_chunk_stride = UInt32(2 * DSTATE)
    var x_n_stride = UInt32(1)
    # out_z, z: (batch, dim, seqlen)
    var out_z_b_stride = UInt32(dim * seqlen)
    var out_z_d_stride = UInt32(seqlen)
    var out_z_t_stride = UInt32(1)
    var z_b_stride = out_z_b_stride
    var z_d_stride = out_z_d_stride
    var z_t_stride = out_z_t_stride
    # dt: (batch, nheads, seqlen)
    var dt_b_stride = UInt32(nheads * seqlen)
    var dt_h_stride = UInt32(seqlen)
    var dt_s_stride = UInt32(1)
    # A: (nheads,)
    var A_stride = UInt32(1)
    # B, C: (batch, ngroups, dstate, seqlen)
    var B_b_stride = UInt32(ngroups * DSTATE * seqlen)
    var B_g_stride = UInt32(DSTATE * seqlen)
    var B_n_stride = UInt32(seqlen)
    var B_t_stride = UInt32(1)
    var C_b_stride = B_b_stride
    var C_g_stride = B_g_stride
    var C_n_stride = B_n_stride
    var C_t_stride = B_t_stride
    # D: (nheads, headdim) or (nheads,)
    var D_h_stride = UInt32(headdim) if D.dim(1) > 0 else UInt32(1)
    var D_p_stride = UInt32(1)
    # 1D strides
    var dt_bias_stride = UInt32(1)
    var rmsnorm_weight_stride = UInt32(1)
    # outproj_weight: (out_dim, dim)
    var outproj_weight_out_stride = UInt32(dim)
    var outproj_weight_in_stride = UInt32(1)
    var outproj_bias_stride = UInt32(1)

    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_int = Int(thread_id)
    if thread_id_int >= total_batch_dim:
        return

    var b = thread_id_int // dim
    var d = thread_id_int % dim

    if b >= batch or d >= dim:
        return

    var h = d // headdim
    var p = d % headdim
    var group_id = h // ngroups if ngroups > 1 else 0
    var width_minus_1 = width - 1
    var z_start = 0
    var xBC_start = dim
    var dt_start = 2 * dim + 2 * ngroups * DSTATE

    # Initialize state for selective scan
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)
    var cum_a = SIMD[DType.float32, MAX_DSTATE](1.0)
    var cum_b = SIMD[DType.float32, MAX_DSTATE](0.0)

    # Pre-load A values
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var A_offset = UInt32(h) * A_stride
        A_vals[n] = (
            Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]() * LOG2E
        )

    var has_D = D.dim(0) > 0
    var D_val = Float32(0.0)
    if has_D:
        if D.dim(1) > 0:
            var D_offset = UInt32(h) * D_h_stride + UInt32(p) * D_p_stride
            D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()
        else:
            var D_offset = UInt32(h) * D_h_stride
            D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

    var has_dt_bias = dt_bias.dim(0) > 0
    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        var bias_offset = UInt32(h) * dt_bias_stride
        dt_bias_val = Scalar[kernel_dtype](dt_bias.ptr[bias_offset]).cast[
            DType.float32
        ]()

    var chunk_idx = 0
    var t_in_chunk = 0

    # Process sequence
    for t in range(seqlen):
        # Step 1: Load z and dt from zxbcdt
        var z_channel = z_start + d
        var z_offset = (
            UInt32(b) * zxbcdt_b_stride
            + UInt32(t) * zxbcdt_s_stride
            + UInt32(z_channel) * zxbcdt_c_stride
        )
        var z_val = Scalar[kernel_dtype](zxbcdt.ptr[z_offset]).cast[
            DType.float32
        ]()

        var dt_channel = dt_start + h
        var dt_offset = (
            UInt32(b) * zxbcdt_b_stride
            + UInt32(t) * zxbcdt_s_stride
            + UInt32(dt_channel) * zxbcdt_c_stride
        )
        var dt_val = Scalar[kernel_dtype](zxbcdt.ptr[dt_offset]).cast[
            DType.float32
        ]()
        dt_val = dt_val + dt_bias_val
        if Bool(Int(delta_softplus) != 0):
            dt_val = softplus(dt_val)

        # Store dt
        var dt_out_offset = (
            UInt32(b) * dt_b_stride
            + UInt32(h) * dt_h_stride
            + UInt32(t) * dt_s_stride
        )
        dt.ptr[dt_out_offset] = Scalar[kernel_dtype](
            dt_val.cast[kernel_dtype]()
        )
        # Step 2: Compute conv for x channel
        var x_channel_in_xBC = d
        var xBC_channel_in_zxbcdt = xBC_start + x_channel_in_xBC

        var conv_sum = Scalar[kernel_dtype](
            conv_bias.ptr[UInt32(x_channel_in_xBC) * conv_bias_stride]
        ).cast[DType.float32]()

        for w in range(width):
            var input_t = t - (width_minus_1 - w)
            if input_t >= 0:
                var xbc_offset = (
                    UInt32(b) * zxbcdt_b_stride
                    + UInt32(input_t) * zxbcdt_s_stride
                    + UInt32(xBC_channel_in_zxbcdt) * zxbcdt_c_stride
                )
                var input_val = Scalar[kernel_dtype](
                    zxbcdt.ptr[xbc_offset]
                ).cast[DType.float32]()
                var weight_offset = (
                    UInt32(x_channel_in_xBC) * conv_weight_c_stride
                    + UInt32(w) * conv_weight_w_stride
                )
                var weight_val = Scalar[kernel_dtype](
                    conv_weight.ptr[weight_offset]
                ).cast[DType.float32]()
                conv_sum = conv_sum + input_val * weight_val

        # Apply SiLU activation
        var x_val = conv_sum / (1.0 + math.exp(-conv_sum))

        # Step 3: Compute B and C for this group
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            # B channel
            var B_channel_in_xBC = dim + group_id * DSTATE + n
            var B_channel_in_zxbcdt = xBC_start + B_channel_in_xBC

            var B_conv_sum = Scalar[kernel_dtype](
                conv_bias.ptr[UInt32(B_channel_in_xBC) * conv_bias_stride]
            ).cast[DType.float32]()
            for w in range(width):
                var input_t = t - (width_minus_1 - w)
                if input_t >= 0:
                    var xbc_offset = (
                        UInt32(b) * zxbcdt_b_stride
                        + UInt32(input_t) * zxbcdt_s_stride
                        + UInt32(B_channel_in_zxbcdt) * zxbcdt_c_stride
                    )
                    var input_val = Scalar[kernel_dtype](
                        zxbcdt.ptr[xbc_offset]
                    ).cast[DType.float32]()
                    var weight_offset = (
                        UInt32(B_channel_in_xBC) * conv_weight_c_stride
                        + UInt32(w) * conv_weight_w_stride
                    )
                    var weight_val = Scalar[kernel_dtype](
                        conv_weight.ptr[weight_offset]
                    ).cast[DType.float32]()
                    B_conv_sum = B_conv_sum + input_val * weight_val
            B_vals[n] = B_conv_sum / (1.0 + exp(-B_conv_sum))  # SiLU

            # Store B
            var B_offset = (
                UInt32(b) * B_b_stride
                + UInt32(group_id) * B_g_stride
                + UInt32(n) * B_n_stride
                + UInt32(t) * B_t_stride
            )
            B.ptr[B_offset] = Scalar[kernel_dtype](
                B_vals[n].cast[kernel_dtype]()
            )
            # C channel
            var C_channel_in_xBC = (
                dim + ngroups * DSTATE + group_id * DSTATE + n
            )
            var C_channel_in_zxbcdt = xBC_start + C_channel_in_xBC

            var C_conv_sum = Scalar[kernel_dtype](
                conv_bias.ptr[UInt32(C_channel_in_xBC) * conv_bias_stride]
            ).cast[DType.float32]()
            for w in range(width):
                var input_t = t - (width_minus_1 - w)
                if input_t >= 0:
                    var xbc_offset = (
                        UInt32(b) * zxbcdt_b_stride
                        + UInt32(input_t) * zxbcdt_s_stride
                        + UInt32(C_channel_in_zxbcdt) * zxbcdt_c_stride
                    )
                    var input_val = Scalar[kernel_dtype](
                        zxbcdt.ptr[xbc_offset]
                    ).cast[DType.float32]()
                    var weight_offset = (
                        UInt32(C_channel_in_xBC) * conv_weight_c_stride
                        + UInt32(w) * conv_weight_w_stride
                    )
                    var weight_val = Scalar[kernel_dtype](
                        conv_weight.ptr[weight_offset]
                    ).cast[DType.float32]()
                    C_conv_sum = C_conv_sum + input_val * weight_val
            C_vals[n] = C_conv_sum / (1.0 + exp(-C_conv_sum))  # SiLU

            # Store C
            var C_offset = (
                UInt32(b) * C_b_stride
                + UInt32(group_id) * C_g_stride
                + UInt32(n) * C_n_stride
                + UInt32(t) * C_t_stride
            )
            C.ptr[C_offset] = Scalar[kernel_dtype](
                C_vals[n].cast[kernel_dtype]()
            )
        # Step 4: Selective scan computation
        var a_t = exp2(dt_val * A_vals)
        var b_t = B_vals * dt_val

        state = state * a_t + b_t
        var ss_output = (state * C_vals).reduce_add()

        cum_b = cum_b * a_t + b_t
        cum_a = cum_a * a_t

        if has_D:
            ss_output += D_val * x_val

        # Step 5: Apply gating and normalization
        var out_val = ss_output
        if has_rmsnorm:
            var rmsnorm_w = Scalar[kernel_dtype](
                rmsnorm_weight.ptr[UInt32(d) * rmsnorm_weight_stride]
            ).cast[DType.float32]()
            var epsilon_val = Scalar[kernel_dtype](epsilon).cast[
                DType.float32
            ]()
            if norm_before_gate:
                var norm_val = (
                    rsqrt(out_val * out_val + epsilon_val) * rmsnorm_w
                )
                out_val = out_val * norm_val * silu(z_val)
            else:
                var gated = out_val * silu(z_val)
                var norm_val = rsqrt(gated * gated + epsilon_val) * rmsnorm_w
                out_val = gated * norm_val
        else:
            out_val = out_val * silu(z_val)

        # Store z and out_z
        var z_out_offset = (
            UInt32(b) * z_b_stride
            + UInt32(d) * z_d_stride
            + UInt32(t) * z_t_stride
        )
        z.ptr[z_out_offset] = Scalar[kernel_dtype](z_val.cast[kernel_dtype]())
        if out_z.dim(0) > 0:
            var out_z_offset = (
                UInt32(b) * out_z_b_stride
                + UInt32(d) * out_z_d_stride
                + UInt32(t) * out_z_t_stride
            )
            out_z.ptr[out_z_offset] = Scalar[kernel_dtype](
                out_val.cast[kernel_dtype]()
            )
        # Step 6: Output projection (if present)
        if has_outproj:
            # Output projection: output[b, t, o] = sum(input[b, t, d] * weight[o, d] for all d) + bias[o]
            # Note: This implementation processes one d at a time and accumulates contributions.
            # For proper thread safety, output should be initialized to bias before this kernel runs,
            # or use a different parallelization strategy (e.g., process (b, t, o) instead of (b, d)).
            var out_dim = output.dim(2)
            for o in range(out_dim):
                # Load weight[o, d]
                var weight_offset = (
                    UInt32(o) * outproj_weight_out_stride
                    + UInt32(d) * outproj_weight_in_stride
                )
                var weight_val = Scalar[kernel_dtype](
                    outproj_weight.ptr[weight_offset]
                ).cast[DType.float32]()

                # Compute contribution: input[b, t, d] * weight[o, d]
                var contribution = out_val * weight_val

                # Get output location
                var out_o_offset = (
                    UInt32(b) * output_b_stride
                    + UInt32(t) * output_s_stride
                    + UInt32(o) * output_c_stride
                )
                # Initialize with bias on first d, then accumulate contributions
                if d == 0:
                    var bias_val = Float32(0.0)
                    if outproj_bias.dim(0) > 0:
                        var bias_offset = UInt32(o) * outproj_bias_stride
                        bias_val = Scalar[kernel_dtype](
                            outproj_bias.ptr[bias_offset]
                        ).cast[DType.float32]()
                    output.ptr[out_o_offset] = Scalar[kernel_dtype](
                        (bias_val + contribution).cast[kernel_dtype]()
                    )
                else:
                    # Read-modify-write: load current value, add contribution, store
                    # Note: This has a race condition when multiple threads write to same location.
                    # For correctness, output should be pre-initialized or use atomic operations.
                    var current_out = Scalar[kernel_dtype](
                        output.ptr[out_o_offset]
                    ).cast[DType.float32]()
                    current_out = current_out + contribution
                    output.ptr[out_o_offset] = Scalar[kernel_dtype](
                        current_out.cast[kernel_dtype]()
                    )

        else:
            # No output projection - store directly
            var out_offset = (
                UInt32(b) * output_b_stride
                + UInt32(t) * output_s_stride
                + UInt32(d) * output_c_stride
            )
            output.ptr[out_offset] = Scalar[kernel_dtype](
                out_val.cast[kernel_dtype]()
            )
        # Check chunk boundary
        t_in_chunk += 1
        var is_chunk_boundary = t_in_chunk == chunk_size
        var is_last_step = t == seqlen - 1

        if is_chunk_boundary or is_last_step:
            comptime for n in range(DSTATE):
                var x_offset_a = UInt32(
                    b * Int(x_b_stride)
                    + d * Int(x_d_stride)
                    + chunk_idx * Int(x_chunk_stride)
                    + (n * 2) * Int(x_n_stride)
                )
                var x_offset_b = UInt32(
                    b * Int(x_b_stride)
                    + d * Int(x_d_stride)
                    + chunk_idx * Int(x_chunk_stride)
                    + (n * 2 + 1) * Int(x_n_stride)
                )
                x.ptr[x_offset_a] = Scalar[kernel_dtype](
                    cum_a[n].cast[kernel_dtype]()
                )
                x.ptr[x_offset_b] = Scalar[kernel_dtype](
                    cum_b[n].cast[kernel_dtype]()
                )
                cum_a[n] = 1.0
                cum_b[n] = 0.0

            if is_chunk_boundary:
                chunk_idx += 1
                t_in_chunk = 0
