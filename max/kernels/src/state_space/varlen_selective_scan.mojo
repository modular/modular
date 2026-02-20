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

"""Variable-length selective scan kernels for Mamba SSM architecture."""

from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from utils.index import IndexList
from memory import UnsafePointer
from algorithm import sync_parallelize
import math
from math import ceildiv, exp2
from state_space.causal_conv1d import silu
from state_space.selective_scan import softplus

# LOG2E constant for converting exp to exp2 (faster on GPU)
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 256  # Larger for Mamba-2 models

# Stride types for passing tensor strides to kernels
comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


fn varlen_selective_state_update_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    state_layout: Layout,
    x_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    out_layout: Layout,
    dt_bias_layout: Layout,
    state_batch_indices_layout: Layout,
](
    # Grid dimensions
    total_threads: Int,  # batch * nheads * dim / BLOCK_SIZE_M
    batch: Int,
    nheads: Int,
    dim: Int,
    nheads_ngroups_ratio: Int,
    pad_slot_id: Int32,
    dt_softplus: Int8,
    has_state_batch_indices: Int8,
    # Tensors
    state: LayoutTensor[kernel_dtype, state_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, out_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, dt_bias_layout, MutAnyOrigin],
    state_batch_indices: LayoutTensor[
        DType.int32, state_batch_indices_layout, MutAnyOrigin
    ],
    state_strides: Strides4D,  # (batch, nheads, dim, dstate)
    x_strides: Strides3D,  # (batch, nheads, dim)
    dt_strides: Strides3D,  # (batch, nheads, dim)
    dt_bias_strides: Strides2D,  # (nheads, dim)
    A_strides: Strides3D,  # (nheads, dim, dstate)
    B_strides: Strides3D,  # (batch, ngroups, dstate)
    C_strides: Strides3D,  # (batch, ngroups, dstate)
    D_strides: Strides2D,  # (nheads, dim)
    z_strides: Strides3D,  # (batch, nheads, dim)
    out_strides: Strides3D,  # (batch, nheads, dim)
):
    """GPU kernel for selective state update with multi-head support."""
    comptime BLOCK_SIZE_M = 4  # Process 4 dims per thread

    var pid_m = block_idx.x  # Dim block index
    var pid_b = block_idx.y  # Batch index
    var pid_h = block_idx.z  # Head index

    var pid_b_int = Int(pid_b)
    var pid_h_int = Int(pid_h)
    var pid_m_int = Int(pid_m)

    if pid_b_int >= batch or pid_h_int >= nheads:
        return

    # Determine state batch index
    var state_batch_idx = Int32(pid_b_int)
    if Bool(Int(has_state_batch_indices) != 0):
        state_batch_idx = state_batch_indices.ptr[pid_b_int]
        # Check for padding
        if state_batch_idx == pad_slot_id:
            return

    var has_dt_bias = dt_bias.dim(0) > 0
    var has_D = D.dim(0) > 0
    var has_z = z.dim(0) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = pid_h_int // nheads_ngroups_ratio

    # Process BLOCK_SIZE_M dims per thread
    comptime for local_m in range(BLOCK_SIZE_M):
        var m = pid_m_int * BLOCK_SIZE_M + local_m
        if m >= dim:
            continue

        # Load x value
        var x_offset = UInt32(
            pid_b_int * x_strides[0]
            + pid_h_int * x_strides[1]
            + m * x_strides[2]
        )
        var x_val = Scalar[kernel_dtype](x.ptr[x_offset]).cast[DType.float32]()

        # Load dt value
        var dt_offset = UInt32(
            pid_b_int * dt_strides[0]
            + pid_h_int * dt_strides[1]
            + m * dt_strides[2]
        )
        var dt_val = Scalar[kernel_dtype](dt.ptr[dt_offset]).cast[
            DType.float32
        ]()

        # Apply dt_bias if present
        if has_dt_bias:
            var dt_bias_offset = UInt32(
                pid_h_int * dt_bias_strides[0] + m * dt_bias_strides[1]
            )
            var bias_val = Scalar[kernel_dtype](
                dt_bias.ptr[dt_bias_offset]
            ).cast[DType.float32]()
            dt_val += bias_val

        # Apply softplus if requested
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var out_val = Float32(0.0)

        # Process each dstate element
        comptime for n in range(DSTATE):
            # Load A value
            var A_offset = UInt32(
                pid_h_int * A_strides[0] + m * A_strides[1] + n * A_strides[2]
            )
            var A_val = Scalar[kernel_dtype](A.ptr[A_offset]).cast[
                DType.float32
            ]()

            # Compute dA = exp(A * dt) using exp2 for faster GPU execution
            var dA = exp2(A_val * LOG2E * dt_val)

            # Load B value
            var B_offset = UInt32(
                pid_b_int * B_strides[0]
                + group_id * B_strides[1]
                + n * B_strides[2]
            )
            var B_val = Scalar[kernel_dtype](B.ptr[B_offset]).cast[
                DType.float32
            ]()

            # Compute dB = B * dt
            var dB = B_val * dt_val

            # Load current state
            var state_offset = UInt32(
                Int(state_batch_idx) * state_strides[0]
                + pid_h_int * state_strides[1]
                + m * state_strides[2]
                + n * state_strides[3]
            )
            var state_val = Scalar[kernel_dtype](state.ptr[state_offset]).cast[
                DType.float32
            ]()

            # Update state: state = state * dA + dB * x
            state_val = state_val * dA + dB * x_val

            # Store updated state
            state.ptr[state_offset] = Scalar[kernel_dtype](
                state_val.cast[kernel_dtype]()
            )

            # Load C value
            var C_offset = UInt32(
                pid_b_int * C_strides[0]
                + group_id * C_strides[1]
                + n * C_strides[2]
            )
            var C_val = Scalar[kernel_dtype](C.ptr[C_offset]).cast[
                DType.float32
            ]()

            # Accumulate output
            out_val += state_val * C_val

        # Add skip connection if D is present
        if has_D:
            var D_offset = UInt32(pid_h_int * D_strides[0] + m * D_strides[1])
            var D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[
                DType.float32
            ]()
            out_val += x_val * D_val

        # Apply gating if z is present, using optimized silu
        if has_z:
            var z_offset = UInt32(
                pid_b_int * z_strides[0]
                + pid_h_int * z_strides[1]
                + m * z_strides[2]
            )
            var z_val = Scalar[kernel_dtype](z.ptr[z_offset]).cast[
                DType.float32
            ]()
            out_val *= silu(z_val)

        # Store output
        var out_offset = UInt32(
            pid_b_int * out_strides[0]
            + pid_h_int * out_strides[1]
            + m * out_strides[2]
        )
        output.ptr[out_offset] = Scalar[kernel_dtype](
            out_val.cast[kernel_dtype]()
        )


fn varlen_selective_scan_fwd_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
    ssm_states_layout: Layout,
    out_layout: Layout,
    query_start_loc_layout: Layout,
    cache_indices_layout: Layout,
    has_initial_state_layout: Layout,
](
    dim: Int,
    ngroups: Int,
    batch: Int,
    pad_slot_id: Int32,
    delta_softplus: Int8,
    # Tensors - varlen format: (dim, total_length) for u, delta, z, out
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[
        kernel_dtype, B_layout, MutAnyOrigin
    ],  # (ngroups, dstate, total_length)
    C: LayoutTensor[
        kernel_dtype, C_layout, MutAnyOrigin
    ],  # (ngroups, dstate, total_length)
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    ssm_states: LayoutTensor[
        kernel_dtype, ssm_states_layout, MutAnyOrigin
    ],  # (batch, dim, dstate)
    output: LayoutTensor[
        kernel_dtype, out_layout, MutAnyOrigin
    ],  # Output written here (or to z if z is present)
    query_start_loc: LayoutTensor[
        DType.int32, query_start_loc_layout, MutAnyOrigin
    ],  # (batch + 1,)
    cache_indices: LayoutTensor[
        DType.int32, cache_indices_layout, MutAnyOrigin
    ],  # (batch,)
    has_initial_state: LayoutTensor[
        DType.bool, has_initial_state_layout, MutAnyOrigin
    ],  # (batch,)
    u_strides: Strides2D,  # (dim, total_length)
    delta_strides: Strides2D,  # (dim, total_length)
    A_strides: Strides2D,  # (dim, dstate)
    B_strides: Strides3D,  # (ngroups, dstate, total_length)
    C_strides: Strides3D,  # (ngroups, dstate, total_length)
    D_strides: Strides1D,  # (dim,)
    z_strides: Strides2D,  # (dim, total_length)
    delta_bias_strides: Strides1D,  # (dim,)
    ssm_states_strides: Strides3D,  # (batch, dim, dstate)
    out_strides: Strides2D,  # (dim, total_length)
):
    """GPU kernel for variable-length selective scan."""
    # 2D grid: block_idx.x for dim, block_idx.y for batch
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    var b = Int(block_idx.y)

    if d >= dim or b >= batch:
        return

    var has_D = D.dim(0) > 0
    var has_z = z.dim(0) > 0
    var has_delta_bias = delta_bias.dim(0) > 0
    var has_cache_indices = cache_indices.dim(0) > 0
    var has_initial_state_tensor = has_initial_state.dim(0) > 0
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)

    # Get sequence start and length
    var seq_start = Int(query_start_loc.ptr[b])
    var seq_end = Int(query_start_loc.ptr[b + 1])
    var seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    # Get cache index for this sequence
    var cache_idx = b
    if has_cache_indices:
        cache_idx = Int(cache_indices.ptr[b])
        if cache_idx == Int(pad_slot_id):
            return

    # Pre-load D and delta_bias for this dim
    var D_val = Float32(0.0)
    if has_D:
        var D_offset = UInt32(d * D_strides[0])
        D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

    var delta_bias_val = Float32(0.0)
    if has_delta_bias:
        var bias_offset = UInt32(d * delta_bias_strides[0])
        delta_bias_val = Scalar[kernel_dtype](delta_bias.ptr[bias_offset]).cast[
            DType.float32
        ]()

    # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
    var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
        A_vals[n] = (
            Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]() * LOG2E
        )

    # Determine group for this dim
    var group_size = dim // ngroups
    var group_id = d // group_size

    # Initialize state - either from cache or zeros
    var state = SIMD[DType.float32, MAX_DSTATE](0.0)

    # Load initial state if requested
    var use_initial_state = False
    if has_initial_state_tensor:
        var init_state_val = has_initial_state.ptr[b]
        use_initial_state = Bool(init_state_val)

    if use_initial_state:
        comptime for n in range(DSTATE):
            var state_offset = UInt32(
                cache_idx * ssm_states_strides[0]
                + d * ssm_states_strides[1]
                + n * ssm_states_strides[2]
            )
            state[n] = Scalar[kernel_dtype](ssm_states.ptr[state_offset]).cast[
                DType.float32
            ]()

    # Process sequence
    for t in range(seq_len):
        var global_t = seq_start + t

        # Load u value
        var u_offset = UInt32(d * u_strides[0] + global_t * u_strides[1])
        var u_val = Scalar[kernel_dtype](u.ptr[u_offset]).cast[DType.float32]()

        # Load delta value
        var delta_offset = UInt32(
            d * delta_strides[0] + global_t * delta_strides[1]
        )
        var delta_val = Scalar[kernel_dtype](delta.ptr[delta_offset]).cast[
            DType.float32
        ]()

        # Apply delta_bias
        if has_delta_bias:
            delta_val += delta_bias_val

        # Apply softplus
        if delta_softplus_bool:
            delta_val = softplus(delta_val)

        var delta_u = delta_val * u_val

        # Load B and C values for this timestep
        var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var B_offset = UInt32(
                group_id * B_strides[0]
                + n * B_strides[1]
                + global_t * B_strides[2]
            )
            var C_offset = UInt32(
                group_id * C_strides[0]
                + n * C_strides[1]
                + global_t * C_strides[2]
            )

            B_vals[n] = Scalar[kernel_dtype](B.ptr[B_offset]).cast[
                DType.float32
            ]()
            C_vals[n] = Scalar[kernel_dtype](C.ptr[C_offset]).cast[
                DType.float32
            ]()

        # SSM step: state = state * exp2(A * LOG2E * delta) + B * delta * u
        var a_t = exp2(A_vals * delta_val)
        var b_t = B_vals * delta_u
        state = state * a_t + b_t

        # Compute output: y = sum(state * C) - use SIMD reduce
        var output_val = (state * C_vals).reduce_add()

        # Add D * u if D is present
        if has_D:
            output_val += D_val * u_val

        # Apply gating with z if present, using optimized silu
        if has_z:
            var z_offset = UInt32(d * z_strides[0] + global_t * z_strides[1])
            var z_val = Scalar[kernel_dtype](z.ptr[z_offset]).cast[
                DType.float32
            ]()
            output_val *= silu(z_val)

            # Write to z if z is present (vLLM convention: output written to z)
            z.ptr[z_offset] = Scalar[kernel_dtype](
                output_val.cast[kernel_dtype]()
            )
        else:
            # Write to output (or delta in vLLM convention)
            var out_offset = UInt32(
                d * out_strides[0] + global_t * out_strides[1]
            )
            output.ptr[out_offset] = Scalar[kernel_dtype](
                output_val.cast[kernel_dtype]()
            )

    # Store final state to cache
    comptime for n in range(DSTATE):
        var state_offset = UInt32(
            cache_idx * ssm_states_strides[0]
            + d * ssm_states_strides[1]
            + n * ssm_states_strides[2]
        )
        ssm_states.ptr[state_offset] = Scalar[kernel_dtype](
            state[n].cast[kernel_dtype]()
        )


fn varlen_selective_state_update_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
    state_layout: Layout,
    x_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    out_layout: Layout,
    dt_bias_layout: Layout,
    state_batch_indices_layout: Layout,
](
    batch: Int,
    nheads: Int,
    dim: Int,
    nheads_ngroups_ratio: Int,
    pad_slot_id: Int32,
    dt_softplus: Int8,
    has_state_batch_indices: Int8,
    # Tensors
    state: LayoutTensor[kernel_dtype, state_layout, MutAnyOrigin],
    x: LayoutTensor[kernel_dtype, x_layout, MutAnyOrigin],
    dt: LayoutTensor[kernel_dtype, dt_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, out_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[kernel_dtype, dt_bias_layout, MutAnyOrigin],
    state_batch_indices: LayoutTensor[
        DType.int32, state_batch_indices_layout, MutAnyOrigin
    ],
    # All strides (same as GPU version)
    state_strides: Strides4D,
    x_strides: Strides3D,
    dt_strides: Strides3D,
    dt_bias_strides: Strides2D,
    A_strides: Strides3D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides2D,
    z_strides: Strides3D,
    out_strides: Strides3D,
):
    """CPU kernel for varlen selective state update."""
    var has_dt_bias = dt_bias.dim(0) > 0
    var has_D = D.dim(0) > 0
    var has_z = z.dim(0) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)
    var has_state_batch_indices_bool = Bool(Int(has_state_batch_indices) != 0)

    @parameter
    fn worker(idx: Int):
        var b = idx // (nheads * dim)
        var remaining = idx % (nheads * dim)
        var h = remaining // dim
        var m = remaining % dim

        # Determine state batch index
        var state_batch_idx = Int32(b)
        if has_state_batch_indices_bool:
            state_batch_idx = state_batch_indices.ptr[b]
            if state_batch_idx == pad_slot_id:
                return

        var group_id = h // nheads_ngroups_ratio

        # Load x value
        var x_offset = UInt32(
            b * x_strides[0] + h * x_strides[1] + m * x_strides[2]
        )
        var x_val = Scalar[kernel_dtype](x.ptr[x_offset]).cast[DType.float32]()

        # Load dt value
        var dt_offset = UInt32(
            b * dt_strides[0] + h * dt_strides[1] + m * dt_strides[2]
        )
        var dt_val = Scalar[kernel_dtype](dt.ptr[dt_offset]).cast[
            DType.float32
        ]()

        # Apply dt_bias if present
        if has_dt_bias:
            var dt_bias_offset = UInt32(
                h * dt_bias_strides[0] + m * dt_bias_strides[1]
            )
            var bias_val = Scalar[kernel_dtype](
                dt_bias.ptr[dt_bias_offset]
            ).cast[DType.float32]()
            dt_val += bias_val

        # Apply softplus if requested
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var out_val = Float32(0.0)

        # Process each dstate element
        comptime for n in range(DSTATE):
            # Load A value
            var A_offset = UInt32(
                h * A_strides[0] + m * A_strides[1] + n * A_strides[2]
            )
            var A_val = Scalar[kernel_dtype](A.ptr[A_offset]).cast[
                DType.float32
            ]()

            # Compute dA = exp(A * dt) using exp2 for consistency
            var dA = exp2(A_val * LOG2E * dt_val)

            # Load B value
            var B_offset = UInt32(
                b * B_strides[0] + group_id * B_strides[1] + n * B_strides[2]
            )
            var B_val = Scalar[kernel_dtype](B.ptr[B_offset]).cast[
                DType.float32
            ]()

            # Compute dB = B * dt
            var dB = B_val * dt_val

            # Load current state
            var state_offset = UInt32(
                Int(state_batch_idx) * state_strides[0]
                + h * state_strides[1]
                + m * state_strides[2]
                + n * state_strides[3]
            )
            var state_val = Scalar[kernel_dtype](state.ptr[state_offset]).cast[
                DType.float32
            ]()

            # Update state
            state_val = state_val * dA + dB * x_val

            # Store updated state
            state.ptr[state_offset] = Scalar[kernel_dtype](
                state_val.cast[kernel_dtype]()
            )

            # Load C value
            var C_offset = UInt32(
                b * C_strides[0] + group_id * C_strides[1] + n * C_strides[2]
            )
            var C_val = Scalar[kernel_dtype](C.ptr[C_offset]).cast[
                DType.float32
            ]()

            # Accumulate output
            out_val += state_val * C_val

        # Add skip connection if D is present
        if has_D:
            var D_offset = UInt32(h * D_strides[0] + m * D_strides[1])
            var D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[
                DType.float32
            ]()
            out_val += x_val * D_val

        # Apply gating if z is present, using optimized silu
        if has_z:
            var z_offset = UInt32(
                b * z_strides[0] + h * z_strides[1] + m * z_strides[2]
            )
            var z_val = Scalar[kernel_dtype](z.ptr[z_offset]).cast[
                DType.float32
            ]()
            out_val *= silu(z_val)

        # Store output
        var out_offset = UInt32(
            b * out_strides[0] + h * out_strides[1] + m * out_strides[2]
        )
        output.ptr[out_offset] = Scalar[kernel_dtype](
            out_val.cast[kernel_dtype]()
        )

    sync_parallelize[worker](batch * nheads * dim)


fn varlen_selective_scan_fwd_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    delta_bias_layout: Layout,
    ssm_states_layout: Layout,
    out_layout: Layout,
    query_start_loc_layout: Layout,
    cache_indices_layout: Layout,
    has_initial_state_layout: Layout,
](
    dim: Int,
    ngroups: Int,
    batch: Int,
    pad_slot_id: Int32,
    delta_softplus: Int8,
    # Tensors
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    delta_bias: LayoutTensor[kernel_dtype, delta_bias_layout, MutAnyOrigin],
    ssm_states: LayoutTensor[kernel_dtype, ssm_states_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, out_layout, MutAnyOrigin],
    query_start_loc: LayoutTensor[
        DType.int32, query_start_loc_layout, MutAnyOrigin
    ],
    cache_indices: LayoutTensor[
        DType.int32, cache_indices_layout, MutAnyOrigin
    ],
    has_initial_state: LayoutTensor[
        DType.bool, has_initial_state_layout, MutAnyOrigin
    ],
    # Strides (same as GPU version)
    u_strides: Strides2D,
    delta_strides: Strides2D,
    A_strides: Strides2D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    z_strides: Strides2D,
    delta_bias_strides: Strides1D,
    ssm_states_strides: Strides3D,
    out_strides: Strides2D,
):
    """CPU kernel for variable-length selective scan."""
    var has_D = D.dim(0) > 0
    var has_z = z.dim(0) > 0
    var has_delta_bias = delta_bias.dim(0) > 0
    var has_cache_indices = cache_indices.dim(0) > 0
    var has_initial_state_tensor = has_initial_state.dim(0) > 0
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    var group_size = dim // ngroups

    @parameter
    fn worker(d: Int):
        # Pre-load D and delta_bias for this dim
        var D_val = Float32(0.0)
        if has_D:
            var D_offset = UInt32(d * D_strides[0])
            D_val = Scalar[kernel_dtype](D.ptr[D_offset]).cast[DType.float32]()

        var delta_bias_val = Float32(0.0)
        if has_delta_bias:
            var bias_offset = UInt32(d * delta_bias_strides[0])
            delta_bias_val = Scalar[kernel_dtype](
                delta_bias.ptr[bias_offset]
            ).cast[DType.float32]()

        # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
        var A_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
            A_vals[n] = (
                Scalar[kernel_dtype](A.ptr[A_offset]).cast[DType.float32]()
                * LOG2E
            )

        var group_id = d // group_size

        # Process each sequence
        for b in range(batch):
            var seq_start = Int(query_start_loc.ptr[b])
            var seq_end = Int(query_start_loc.ptr[b + 1])
            var seq_len = seq_end - seq_start

            if seq_len <= 0:
                continue

            var cache_idx = b
            if has_cache_indices:
                cache_idx = Int(cache_indices.ptr[b])
                if cache_idx == Int(pad_slot_id):
                    continue

            # Initialize state
            var state = SIMD[DType.float32, MAX_DSTATE](0.0)

            var use_initial_state = False
            if has_initial_state_tensor:
                var init_state_val = has_initial_state.ptr[b]
                use_initial_state = Bool(init_state_val)

            if use_initial_state:
                comptime for n in range(DSTATE):
                    var state_offset = UInt32(
                        cache_idx * ssm_states_strides[0]
                        + d * ssm_states_strides[1]
                        + n * ssm_states_strides[2]
                    )
                    state[n] = Scalar[kernel_dtype](
                        ssm_states.ptr[state_offset]
                    ).cast[DType.float32]()

            # Process sequence
            for t in range(seq_len):
                var global_t = seq_start + t

                var u_offset = UInt32(
                    d * u_strides[0] + global_t * u_strides[1]
                )
                var u_val = Scalar[kernel_dtype](u.ptr[u_offset]).cast[
                    DType.float32
                ]()

                var delta_offset = UInt32(
                    d * delta_strides[0] + global_t * delta_strides[1]
                )
                var out_offset = UInt32(
                    d * out_strides[0] + global_t * out_strides[1]
                )
                var delta_val = Scalar[kernel_dtype](
                    delta.ptr[delta_offset]
                ).cast[DType.float32]()

                if has_delta_bias:
                    delta_val += delta_bias_val

                if delta_softplus_bool:
                    delta_val = softplus(delta_val)

                var delta_u = delta_val * u_val

                var B_vals = SIMD[DType.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[DType.float32, MAX_DSTATE](0.0)

                comptime for n in range(DSTATE):
                    var B_offset = UInt32(
                        group_id * B_strides[0]
                        + n * B_strides[1]
                        + global_t * B_strides[2]
                    )
                    var C_offset = UInt32(
                        group_id * C_strides[0]
                        + n * C_strides[1]
                        + global_t * C_strides[2]
                    )

                    B_vals[n] = Scalar[kernel_dtype](B.ptr[B_offset]).cast[
                        DType.float32
                    ]()
                    C_vals[n] = Scalar[kernel_dtype](C.ptr[C_offset]).cast[
                        DType.float32
                    ]()

                # SSM step using SIMD exp2 with pre-multiplied LOG2E
                var a_t = exp2(A_vals * delta_val)
                var b_t = B_vals * delta_u
                state = state * a_t + b_t

                # Compute output using SIMD reduce
                var output_val = (state * C_vals).reduce_add()

                if has_D:
                    output_val += D_val * u_val

                if has_z:
                    var z_offset = UInt32(
                        d * z_strides[0] + global_t * z_strides[1]
                    )
                    var z_val = Scalar[kernel_dtype](z.ptr[z_offset]).cast[
                        DType.float32
                    ]()
                    output_val *= silu(z_val)
                    z.ptr[z_offset] = Scalar[kernel_dtype](
                        output_val.cast[kernel_dtype]()
                    )
                else:
                    output.ptr[out_offset] = Scalar[kernel_dtype](
                        output_val.cast[kernel_dtype]()
                    )

            # Store final state
            comptime for n in range(DSTATE):
                var state_offset = UInt32(
                    cache_idx * ssm_states_strides[0]
                    + d * ssm_states_strides[1]
                    + n * ssm_states_strides[2]
                )
                ssm_states.ptr[state_offset] = Scalar[kernel_dtype](
                    state[n].cast[kernel_dtype]()
                )

    sync_parallelize[worker](dim)
