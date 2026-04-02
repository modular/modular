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
"""Varlen selective scan operation registrations for Mamba SSM.

This module registers operations for variable-length selective scan:
- varlen_selective_scan_fwd: Forward pass for varlen sequences
- varlen_selective_state_update: State update for decode/autoregressive inference
"""

from std.math import ceildiv

import compiler_internal as compiler
from std.gpu.host.info import is_cpu, is_gpu
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

from state_space.varlen_selective_scan import (
    varlen_selective_scan_fwd_cpu,
    varlen_selective_scan_fwd_gpu,
    varlen_selective_state_update_cpu,
    varlen_selective_state_update_gpu,
)

# Stride types for kernel calls
comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


# ============================================================================
# Varlen Selective Scan Forward Registration
# ============================================================================


@compiler.register("varlen_selective_scan_fwd")
struct VarlenSelectiveScanFwd[delta_softplus: Bool = False]:
    """Variable-length selective scan forward pass.

    Performs the selective scan computation for variable-length sequences
    that are concatenated together. Uses cumulative sequence lengths to
    identify sequence boundaries.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (dim, total_length) - Output tensor (or written to z if present)
        - ssm_states: (batch, dim, dstate) - SSM states (in/out)
        - u: (dim, total_length) - Input tensor
        - delta: (dim, total_length) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (ngroups, dstate, total_length) - Input projection
        - C: (ngroups, dstate, total_length) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (dim, total_length) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
        - query_start_loc: (batch + 1,) - Cumulative sequence lengths
        - cache_indices: (batch,) - Indices into ssm_states (optional)
        - has_initial_state: (batch,) - Whether to use initial state (optional)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        ssm_states: OutputTensor[dtype=dtype, rank=3, ...],
        z: OutputTensor[dtype=dtype, rank=2, ...],
        u: InputTensor[dtype=dtype, rank=2, ...],
        delta: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
        query_start_loc: InputTensor[dtype=DType.int32, rank=1, ...],
        cache_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        has_initial_state: InputTensor[dtype=DType.bool, rank=1, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var dim = u.dim_size(0)
        var dstate = A.dim_size(1)
        var ngroups = B.dim_size(0)
        var batch = query_start_loc.dim_size(0) - 1

        var output_tt = output.to_tile_tensor[DType.int32]()
        var ssm_states_tt = ssm_states.to_tile_tensor[DType.int32]()
        var u_tt = u.to_tile_tensor[DType.int32]()
        var delta_tt = delta.to_tile_tensor[DType.int32]()
        var A_tt = A.to_tile_tensor[DType.int32]()
        var B_tt = B.to_tile_tensor[DType.int32]()
        var C_tt = C.to_tile_tensor[DType.int32]()
        var D_tt = D.to_tile_tensor[DType.int32]()
        var z_tt = z.to_tile_tensor[DType.int32]()
        var delta_bias_tt = delta_bias.to_tile_tensor[DType.int32]()
        var query_start_loc_tt = query_start_loc.to_tile_tensor[DType.int32]()
        var cache_indices_tt = cache_indices.to_tile_tensor[DType.int32]()
        var has_initial_state_tt = has_initial_state.to_tile_tensor[
            DType.int32
        ]()

        # Get strides
        var u_strides = Strides2D(u.strides()[0], u.strides()[1])
        var delta_strides = Strides2D(delta.strides()[0], delta.strides()[1])
        var A_strides = Strides2D(A.strides()[0], A.strides()[1])
        var B_strides = Strides3D(
            B.strides()[0], B.strides()[1], B.strides()[2]
        )
        var C_strides = Strides3D(
            C.strides()[0], C.strides()[1], C.strides()[2]
        )
        var D_strides = Strides1D(D.strides()[0] if D.dim_size(0) > 0 else 1)
        var z_strides = Strides2D(
            z.strides()[0] if z.dim_size(0) > 0 else 1,
            z.strides()[1] if z.dim_size(0) > 0 else 1,
        )
        var delta_bias_strides = Strides1D(
            delta_bias.strides()[0] if delta_bias.dim_size(0) > 0 else 1
        )
        var ssm_states_strides = Strides3D(
            ssm_states.strides()[0],
            ssm_states.strides()[1],
            ssm_states.strides()[2],
        )
        var out_strides = Strides2D(output.strides()[0], output.strides()[1])

        comptime PAD_SLOT_ID: Int32 = -1
        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 4 and dstate != 8 and dstate != 16:
            raise Error(
                "Unsupported dstate: "
                + String(dstate)
                + ". Expected 4, 8, or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                varlen_selective_scan_fwd_cpu[
                    dtype,
                    16,
                ](
                    dim,
                    ngroups,
                    batch,
                    PAD_SLOT_ID,
                    delta_softplus_int8,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    ssm_states_tt,
                    output_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    ssm_states_strides,
                    out_strides,
                )
            elif dstate == 8:
                varlen_selective_scan_fwd_cpu[
                    dtype,
                    8,
                ](
                    dim,
                    ngroups,
                    batch,
                    PAD_SLOT_ID,
                    delta_softplus_int8,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    ssm_states_tt,
                    output_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    ssm_states_strides,
                    out_strides,
                )
            else:  # dstate == 4
                varlen_selective_scan_fwd_cpu[
                    dtype,
                    4,
                ](
                    dim,
                    ngroups,
                    batch,
                    PAD_SLOT_ID,
                    delta_softplus_int8,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    ssm_states_tt,
                    output_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    ssm_states_strides,
                    out_strides,
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            comptime BLOCK_SIZE = 128
            var num_dim_blocks = ceildiv(dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    varlen_selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        ssm_states_tt.LayoutType,
                        output_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                    ],
                    varlen_selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        ssm_states_tt.LayoutType,
                        output_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    dim,
                    ngroups,
                    batch,
                    PAD_SLOT_ID,
                    delta_softplus_int8,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    ssm_states_tt,
                    output_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    ssm_states_strides,
                    out_strides,
                    grid_dim=(num_dim_blocks, batch, 1),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
            elif dstate == 8:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    varlen_selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        ssm_states_tt.LayoutType,
                        output_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                    ],
                    varlen_selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        ssm_states_tt.LayoutType,
                        output_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    dim,
                    ngroups,
                    batch,
                    PAD_SLOT_ID,
                    delta_softplus_int8,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    ssm_states_tt,
                    output_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    ssm_states_strides,
                    out_strides,
                    grid_dim=(num_dim_blocks, batch, 1),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
            else:  # dstate == 4
                comptime DSTATE_VAL = 4
                var compiled_kernel = gpu_ctx.compile_function[
                    varlen_selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        ssm_states_tt.LayoutType,
                        output_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                    ],
                    varlen_selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        ssm_states_tt.LayoutType,
                        output_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    dim,
                    ngroups,
                    batch,
                    PAD_SLOT_ID,
                    delta_softplus_int8,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    ssm_states_tt,
                    output_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    ssm_states_strides,
                    out_strides,
                    grid_dim=(num_dim_blocks, batch, 1),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    def shape[
        dtype: DType,
    ](
        u: InputTensor[dtype=dtype, rank=2, ...],
        delta: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
        query_start_loc: InputTensor[dtype=DType.int32, rank=1, ...],
        cache_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        has_initial_state: InputTensor[dtype=DType.bool, rank=1, ...],
    ) -> IndexList[2]:
        return u.shape()


# ============================================================================
# Varlen Selective State Update Registration
# ============================================================================


@compiler.register("varlen_selective_state_update")
struct VarlenSelectiveStateUpdate[dt_softplus: Bool = False]:
    """Varlen selective state update for autoregressive inference.

    Performs a single step of the SSM recurrence for incremental token
    generation with multi-head support.

    Parameters:
        dt_softplus: If True, applies softplus activation to dt values.

    Tensor Shapes:
        - state: (batch, nheads, dim, dstate) - SSM state (in/out)
        - output: (batch, nheads, dim) - Output tensor
        - x: (batch, nheads, dim) - Input tensor
        - dt: (batch, nheads, dim) - Time delta tensor
        - A: (nheads, dim, dstate) - State transition matrix
        - B: (batch, ngroups, dstate) - Input matrix
        - C: (batch, ngroups, dstate) - Output matrix
        - D: (nheads, dim) - Skip connection (optional, can be empty)
        - z: (batch, nheads, dim) - Gating tensor (optional, can be empty)
        - dt_bias: (nheads, dim) - Time delta bias (optional, can be empty)
        - state_batch_indices: (batch,) - Indices into state batch (optional)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        state: OutputTensor[dtype=dtype, rank=4, ...],
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        dt: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=3, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=2, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        dt_bias: InputTensor[dtype=dtype, rank=2, ...],
        state_batch_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var batch = x.dim_size(0)
        var nheads = x.dim_size(1)
        var dim = x.dim_size(2)
        var dstate = state.dim_size(3)
        var ngroups = B.dim_size(1)
        var nheads_ngroups_ratio = nheads // ngroups

        var state_tt = state.to_tile_tensor[DType.int32]()
        var output_tt = output.to_tile_tensor[DType.int32]()
        var x_tt = x.to_tile_tensor[DType.int32]()
        var dt_tt = dt.to_tile_tensor[DType.int32]()
        var A_tt = A.to_tile_tensor[DType.int32]()
        var B_tt = B.to_tile_tensor[DType.int32]()
        var C_tt = C.to_tile_tensor[DType.int32]()
        var D_tt = D.to_tile_tensor[DType.int32]()
        var z_tt = z.to_tile_tensor[DType.int32]()
        var dt_bias_tt = dt_bias.to_tile_tensor[DType.int32]()
        var state_batch_indices_tt = state_batch_indices.to_tile_tensor[
            DType.int32
        ]()

        # Get strides
        var state_strides = Strides4D(
            state.strides()[0],
            state.strides()[1],
            state.strides()[2],
            state.strides()[3],
        )
        var x_strides = Strides3D(
            x.strides()[0], x.strides()[1], x.strides()[2]
        )
        var dt_strides = Strides3D(
            dt.strides()[0], dt.strides()[1], dt.strides()[2]
        )
        var dt_bias_strides = Strides2D(
            dt_bias.strides()[0] if dt_bias.dim_size(0) > 0 else 1,
            dt_bias.strides()[1] if dt_bias.dim_size(0) > 0 else 1,
        )
        var A_strides = Strides3D(
            A.strides()[0], A.strides()[1], A.strides()[2]
        )
        var B_strides = Strides3D(
            B.strides()[0], B.strides()[1], B.strides()[2]
        )
        var C_strides = Strides3D(
            C.strides()[0], C.strides()[1], C.strides()[2]
        )
        var D_strides = Strides2D(
            D.strides()[0] if D.dim_size(0) > 0 else 1,
            D.strides()[1] if D.dim_size(0) > 0 else 1,
        )
        var z_strides = Strides3D(
            z.strides()[0] if z.dim_size(0) > 0 else 1,
            z.strides()[1] if z.dim_size(0) > 0 else 1,
            z.strides()[2] if z.dim_size(0) > 0 else 1,
        )
        var out_strides = Strides3D(
            output.strides()[0], output.strides()[1], output.strides()[2]
        )

        var has_state_batch_indices = state_batch_indices.dim_size(0) > 0
        comptime PAD_SLOT_ID: Int32 = -1
        comptime dt_softplus_int8: Int8 = Int8(1) if Self.dt_softplus else Int8(
            0
        )

        if dstate != 4 and dstate != 8 and dstate != 16:
            raise Error(
                "Unsupported dstate: "
                + String(dstate)
                + ". Expected 4, 8, or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                varlen_selective_state_update_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    nheads,
                    dim,
                    nheads_ngroups_ratio,
                    PAD_SLOT_ID,
                    dt_softplus_int8,
                    Int8(has_state_batch_indices),
                    state_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    output_tt,
                    dt_bias_tt,
                    state_batch_indices_tt,
                    state_strides,
                    x_strides,
                    dt_strides,
                    dt_bias_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    out_strides,
                )
            elif dstate == 8:
                varlen_selective_state_update_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    nheads,
                    dim,
                    nheads_ngroups_ratio,
                    PAD_SLOT_ID,
                    dt_softplus_int8,
                    Int8(has_state_batch_indices),
                    state_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    output_tt,
                    dt_bias_tt,
                    state_batch_indices_tt,
                    state_strides,
                    x_strides,
                    dt_strides,
                    dt_bias_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    out_strides,
                )
            else:  # dstate == 4
                varlen_selective_state_update_cpu[
                    dtype,
                    4,
                ](
                    batch,
                    nheads,
                    dim,
                    nheads_ngroups_ratio,
                    PAD_SLOT_ID,
                    dt_softplus_int8,
                    Int8(has_state_batch_indices),
                    state_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    output_tt,
                    dt_bias_tt,
                    state_batch_indices_tt,
                    state_strides,
                    x_strides,
                    dt_strides,
                    dt_bias_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    out_strides,
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            comptime BLOCK_SIZE_M = 4
            var total_threads = batch * nheads * ceildiv(dim, BLOCK_SIZE_M)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    varlen_selective_state_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        output_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_batch_indices_tt.LayoutType,
                    ],
                    varlen_selective_state_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        output_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_batch_indices_tt.LayoutType,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_threads,
                    batch,
                    nheads,
                    dim,
                    nheads_ngroups_ratio,
                    PAD_SLOT_ID,
                    dt_softplus_int8,
                    Int8(has_state_batch_indices),
                    state_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    output_tt,
                    dt_bias_tt,
                    state_batch_indices_tt,
                    state_strides,
                    x_strides,
                    dt_strides,
                    dt_bias_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    out_strides,
                    grid_dim=(ceildiv(dim, BLOCK_SIZE_M), batch, nheads),
                    block_dim=(1,),
                )
            elif dstate == 8:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    varlen_selective_state_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        output_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_batch_indices_tt.LayoutType,
                    ],
                    varlen_selective_state_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        output_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_batch_indices_tt.LayoutType,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_threads,
                    batch,
                    nheads,
                    dim,
                    nheads_ngroups_ratio,
                    PAD_SLOT_ID,
                    dt_softplus_int8,
                    Int8(has_state_batch_indices),
                    state_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    output_tt,
                    dt_bias_tt,
                    state_batch_indices_tt,
                    state_strides,
                    x_strides,
                    dt_strides,
                    dt_bias_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    out_strides,
                    grid_dim=(ceildiv(dim, BLOCK_SIZE_M), batch, nheads),
                    block_dim=(1,),
                )
            else:  # dstate == 4
                comptime DSTATE_VAL = 4
                var compiled_kernel = gpu_ctx.compile_function[
                    varlen_selective_state_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        output_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_batch_indices_tt.LayoutType,
                    ],
                    varlen_selective_state_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        output_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_batch_indices_tt.LayoutType,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_threads,
                    batch,
                    nheads,
                    dim,
                    nheads_ngroups_ratio,
                    PAD_SLOT_ID,
                    dt_softplus_int8,
                    Int8(has_state_batch_indices),
                    state_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    output_tt,
                    dt_bias_tt,
                    state_batch_indices_tt,
                    state_strides,
                    x_strides,
                    dt_strides,
                    dt_bias_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    out_strides,
                    grid_dim=(ceildiv(dim, BLOCK_SIZE_M), batch, nheads),
                    block_dim=(1,),
                )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    def shape[
        dtype: DType,
    ](
        x: InputTensor[dtype=dtype, rank=3, ...],
        dt: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=3, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=2, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        dt_bias: InputTensor[dtype=dtype, rank=2, ...],
        state_batch_indices: InputTensor[dtype=DType.int32, rank=1, ...],
    ) -> Tuple[IndexList[4], IndexList[3]]:
        var batch = x.dim_size(0)
        var nheads = x.dim_size(1)
        var dim = x.dim_size(2)
        var dstate = A.dim_size(2)
        return (IndexList[4](batch, nheads, dim, dstate), x.shape())
