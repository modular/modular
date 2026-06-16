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

import extensibility as compiler
from std.gpu.host import DeviceContext, Dim
from std.gpu.host.info import is_cpu, is_gpu

from extensibility import InputTensor, OutputTensor
from layout import TensorLayout, TileTensor
from std.utils.index import IndexList

from state_space.varlen_selective_scan import (
    Strides1D,
    Strides2D,
    Strides3D,
    Strides4D,
    varlen_selective_scan_fwd_cpu,
    varlen_selective_scan_fwd_gpu,
    varlen_selective_state_update_cpu,
    varlen_selective_state_update_gpu,
)

comptime _GPU_FWD_BLOCK_SIZE = 128
comptime _GPU_UPDATE_BLOCK_SIZE_M = 4
comptime _PAD_SLOT_ID: Int32 = -1

# Supported d_state values for the varlen kernels, largest first (the varlen
# path also supports d_state=4). Single source of truth: _validate_d_state and
# every dispatch_for_d_state below iterate it. Keep _SUPPORTED_D_STATE_VALUES
# (the human-readable error text) in sync with it.
comptime _SUPPORTED_D_STATE = [256, 128, 64, 32, 16, 8, 4]
comptime _SUPPORTED_D_STATE_VALUES = "4, 8, 16, 32, 64, 128, or 256"


def _unsupported_d_state_error(d_state: Int) -> Error:
    return Error(
        "Unsupported d_state: "
        + String(d_state)
        + ". Expected "
        + _SUPPORTED_D_STATE_VALUES
        + "."
    )


def _validate_d_state(d_state: Int) raises:
    comptime for ds in _SUPPORTED_D_STATE:
        if d_state == ds:
            return
    raise _unsupported_d_state_error(d_state)


@fieldwise_init
struct VarlenSelectiveScanFwdStrides(Copyable, ImplicitlyCopyable, Movable):
    var u: Strides2D
    var delta: Strides2D
    var A: Strides2D
    var B: Strides3D
    var C: Strides3D
    var D: Strides1D
    var z: Strides2D
    var delta_bias: Strides1D
    var ssm_states: Strides3D
    var output: Strides2D


@fieldwise_init
struct VarlenSelectiveStateUpdateStrides(Copyable, ImplicitlyCopyable, Movable):
    var state: Strides4D
    var x: Strides3D
    var dt: Strides3D
    var dt_bias: Strides2D
    var A: Strides3D
    var B: Strides3D
    var C: Strides3D
    var D: Strides2D
    var z: Strides3D
    var output: Strides3D


@fieldwise_init
struct VarlenSelectiveScanFwdArgs[
    dtype: DType,
    output_LT: TensorLayout,
    ssm_states_LT: TensorLayout,
    z_LT: TensorLayout,
    u_LT: TensorLayout,
    delta_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    delta_bias_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
]:
    var ctx: DeviceContext
    var dim: Int
    var ngroups: Int
    var batch: Int
    var pad_slot_id: Int32
    var delta_softplus_int8: Int8
    var output_tt: TileTensor[
        mut=True, Self.dtype, Self.output_LT, MutUntrackedOrigin
    ]
    var ssm_states_tt: TileTensor[
        mut=True, Self.dtype, Self.ssm_states_LT, MutUntrackedOrigin
    ]
    var z_tt: TileTensor[mut=True, Self.dtype, Self.z_LT, MutUntrackedOrigin]
    var u_tt: TileTensor[Self.dtype, Self.u_LT, MutUntrackedOrigin]
    var delta_tt: TileTensor[Self.dtype, Self.delta_LT, MutUntrackedOrigin]
    var A_tt: TileTensor[Self.dtype, Self.A_LT, MutUntrackedOrigin]
    var B_tt: TileTensor[Self.dtype, Self.B_LT, MutUntrackedOrigin]
    var C_tt: TileTensor[Self.dtype, Self.C_LT, MutUntrackedOrigin]
    var D_tt: TileTensor[Self.dtype, Self.D_LT, MutUntrackedOrigin]
    var delta_bias_tt: TileTensor[
        Self.dtype, Self.delta_bias_LT, MutUntrackedOrigin
    ]
    var query_start_loc_tt: TileTensor[
        DType.int32, Self.query_start_loc_LT, MutUntrackedOrigin
    ]
    var cache_indices_tt: TileTensor[
        DType.int32, Self.cache_indices_LT, MutUntrackedOrigin
    ]
    var has_initial_state_tt: TileTensor[
        DType.bool, Self.has_initial_state_LT, MutUntrackedOrigin
    ]
    var strides: VarlenSelectiveScanFwdStrides
    var grid_dim: Dim
    var block_dim: Dim

    @parameter
    def launch_gpu[d_state_val: Int](self) capturing raises:
        comptime kernel = varlen_selective_scan_fwd_gpu[
            Self.dtype,
            d_state_val,
            Self.u_LT,
            Self.delta_LT,
            Self.A_LT,
            Self.B_LT,
            Self.C_LT,
            Self.D_LT,
            Self.z_LT,
            Self.delta_bias_LT,
            Self.ssm_states_LT,
            Self.output_LT,
            Self.query_start_loc_LT,
            Self.cache_indices_LT,
            Self.has_initial_state_LT,
        ]
        var compiled_kernel = self.ctx.compile_function[kernel]()
        self.ctx.enqueue_function(
            compiled_kernel,
            self.dim,
            self.ngroups,
            self.batch,
            self.pad_slot_id,
            self.delta_softplus_int8,
            self.u_tt,
            self.delta_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.delta_bias_tt,
            self.ssm_states_tt,
            self.output_tt,
            self.query_start_loc_tt,
            self.cache_indices_tt,
            self.has_initial_state_tt,
            self.strides.u,
            self.strides.delta,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.delta_bias,
            self.strides.ssm_states,
            self.strides.output,
            grid_dim=self.grid_dim,
            block_dim=self.block_dim,
        )

    @parameter
    def run_cpu[d_state_val: Int](self) capturing raises:
        varlen_selective_scan_fwd_cpu[
            Self.dtype,
            d_state_val,
        ](
            self.dim,
            self.ngroups,
            self.batch,
            self.pad_slot_id,
            self.delta_softplus_int8,
            self.u_tt,
            self.delta_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.delta_bias_tt,
            self.ssm_states_tt,
            self.output_tt,
            self.query_start_loc_tt,
            self.cache_indices_tt,
            self.has_initial_state_tt,
            self.strides.u,
            self.strides.delta,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.delta_bias,
            self.strides.ssm_states,
            self.strides.output,
            Optional[DeviceContext](self.ctx),
        )

    def dispatch_for_d_state[
        target: StaticString
    ](mut self, d_state: Int) capturing raises:
        comptime if is_cpu[target]():
            comptime for ds in _SUPPORTED_D_STATE:
                if d_state == ds:
                    self.run_cpu[ds]()
                    return
            raise _unsupported_d_state_error(d_state)
        elif is_gpu[target]():
            comptime for ds in _SUPPORTED_D_STATE:
                if d_state == ds:
                    self.launch_gpu[ds]()
                    return
            raise _unsupported_d_state_error(d_state)
        else:
            raise Error("Unsupported target: " + target)


@fieldwise_init
struct VarlenSelectiveStateUpdateArgs[
    dtype: DType,
    state_LT: TensorLayout,
    output_LT: TensorLayout,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    z_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    state_batch_indices_LT: TensorLayout,
]:
    var ctx: DeviceContext
    var total_threads: Int
    var batch: Int
    var nheads: Int
    var dim: Int
    var nheads_ngroups_ratio: Int
    var pad_slot_id: Int32
    var dt_softplus_int8: Int8
    var has_state_batch_indices_int8: Int8
    var state_tt: TileTensor[
        mut=True, Self.dtype, Self.state_LT, MutUntrackedOrigin
    ]
    var output_tt: TileTensor[
        mut=True, Self.dtype, Self.output_LT, MutUntrackedOrigin
    ]
    var x_tt: TileTensor[Self.dtype, Self.x_LT, MutUntrackedOrigin]
    var dt_tt: TileTensor[Self.dtype, Self.dt_LT, MutUntrackedOrigin]
    var A_tt: TileTensor[Self.dtype, Self.A_LT, MutUntrackedOrigin]
    var B_tt: TileTensor[Self.dtype, Self.B_LT, MutUntrackedOrigin]
    var C_tt: TileTensor[Self.dtype, Self.C_LT, MutUntrackedOrigin]
    var D_tt: TileTensor[Self.dtype, Self.D_LT, MutUntrackedOrigin]
    var z_tt: TileTensor[Self.dtype, Self.z_LT, MutUntrackedOrigin]
    var dt_bias_tt: TileTensor[Self.dtype, Self.dt_bias_LT, MutUntrackedOrigin]
    var state_batch_indices_tt: TileTensor[
        DType.int32, Self.state_batch_indices_LT, MutUntrackedOrigin
    ]
    var strides: VarlenSelectiveStateUpdateStrides
    var grid_dim: Dim
    var block_dim: Dim

    @parameter
    def launch_gpu[d_state_val: Int](self) capturing raises:
        comptime kernel = varlen_selective_state_update_gpu[
            Self.dtype,
            d_state_val,
            Self.state_LT,
            Self.x_LT,
            Self.dt_LT,
            Self.A_LT,
            Self.B_LT,
            Self.C_LT,
            Self.D_LT,
            Self.z_LT,
            Self.output_LT,
            Self.dt_bias_LT,
            Self.state_batch_indices_LT,
        ]
        var compiled_kernel = self.ctx.compile_function[kernel]()
        self.ctx.enqueue_function(
            compiled_kernel,
            self.total_threads,
            self.batch,
            self.nheads,
            self.dim,
            self.nheads_ngroups_ratio,
            self.pad_slot_id,
            self.dt_softplus_int8,
            self.has_state_batch_indices_int8,
            self.state_tt,
            self.x_tt,
            self.dt_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.output_tt,
            self.dt_bias_tt,
            self.state_batch_indices_tt,
            self.strides.state,
            self.strides.x,
            self.strides.dt,
            self.strides.dt_bias,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.output,
            grid_dim=self.grid_dim,
            block_dim=self.block_dim,
        )

    @parameter
    def run_cpu[d_state_val: Int](self) capturing raises:
        varlen_selective_state_update_cpu[
            Self.dtype,
            d_state_val,
        ](
            self.batch,
            self.nheads,
            self.dim,
            self.nheads_ngroups_ratio,
            self.pad_slot_id,
            self.dt_softplus_int8,
            self.has_state_batch_indices_int8,
            self.state_tt,
            self.x_tt,
            self.dt_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.output_tt,
            self.dt_bias_tt,
            self.state_batch_indices_tt,
            self.strides.state,
            self.strides.x,
            self.strides.dt,
            self.strides.dt_bias,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.output,
            Optional[DeviceContext](self.ctx),
        )

    def dispatch_for_d_state[
        target: StaticString
    ](mut self, d_state: Int) capturing raises:
        comptime if is_cpu[target]():
            comptime for ds in _SUPPORTED_D_STATE:
                if d_state == ds:
                    self.run_cpu[ds]()
                    return
            raise _unsupported_d_state_error(d_state)
        elif is_gpu[target]():
            comptime for ds in _SUPPORTED_D_STATE:
                if d_state == ds:
                    self.launch_gpu[ds]()
                    return
            raise _unsupported_d_state_error(d_state)
        else:
            raise Error("Unsupported target: " + target)


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
        - ssm_states: (batch, dim, d_state) - SSM states (in/out)
        - u: (dim, total_length) - Input tensor
        - delta: (dim, total_length) - Time step tensor
        - A: (dim, d_state) - State transition matrix
        - B: (ngroups, d_state, total_length) - Input projection
        - C: (ngroups, d_state, total_length) - Output projection
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
        ctx: DeviceContext,
    ) capturing raises:
        var dim = u.dim_size(0)
        var d_state = A.dim_size(1)
        var ngroups = B.dim_size(0)
        var batch = query_start_loc.dim_size(0) - 1

        var output_tt = output.to_tile_tensor()
        var ssm_states_tt = ssm_states.to_tile_tensor()
        var z_tt = z.to_tile_tensor()
        var u_tt = u.to_tile_tensor()
        var delta_tt = delta.to_tile_tensor()
        var A_tt = A.to_tile_tensor()
        var B_tt = B.to_tile_tensor()
        var C_tt = C.to_tile_tensor()
        var D_tt = D.to_tile_tensor()
        var delta_bias_tt = delta_bias.to_tile_tensor()
        var query_start_loc_tt = query_start_loc.to_tile_tensor()
        var cache_indices_tt = cache_indices.to_tile_tensor()
        var has_initial_state_tt = has_initial_state.to_tile_tensor()

        var strides = VarlenSelectiveScanFwdStrides(
            u=Strides2D(u.strides()[0], u.strides()[1]),
            delta=Strides2D(delta.strides()[0], delta.strides()[1]),
            A=Strides2D(A.strides()[0], A.strides()[1]),
            B=Strides3D(B.strides()[0], B.strides()[1], B.strides()[2]),
            C=Strides3D(C.strides()[0], C.strides()[1], C.strides()[2]),
            D=Strides1D(D.strides()[0] if D.dim_size(0) > 0 else 1),
            z=Strides2D(
                z.strides()[0] if z.dim_size(0) > 0 else 1,
                z.strides()[1] if z.dim_size(0) > 0 else 1,
            ),
            delta_bias=Strides1D(
                delta_bias.strides()[0] if delta_bias.dim_size(0) > 0 else 1
            ),
            ssm_states=Strides3D(
                ssm_states.strides()[0],
                ssm_states.strides()[1],
                ssm_states.strides()[2],
            ),
            output=Strides2D(output.strides()[0], output.strides()[1]),
        )

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        _validate_d_state(d_state)

        var grid_dim: Dim
        var block_dim: Dim
        comptime if is_gpu[target]():
            var num_dim_blocks = ceildiv(dim, _GPU_FWD_BLOCK_SIZE)
            grid_dim = (num_dim_blocks, batch, 1)
            block_dim = (_GPU_FWD_BLOCK_SIZE, 1, 1)
        else:
            grid_dim = (1, 1, 1)
            block_dim = (1, 1, 1)

        var args = VarlenSelectiveScanFwdArgs[
            dtype,
            output_tt.LayoutType,
            ssm_states_tt.LayoutType,
            z_tt.LayoutType,
            u_tt.LayoutType,
            delta_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
            D_tt.LayoutType,
            delta_bias_tt.LayoutType,
            query_start_loc_tt.LayoutType,
            cache_indices_tt.LayoutType,
            has_initial_state_tt.LayoutType,
        ](
            ctx=ctx,
            dim=dim,
            ngroups=ngroups,
            batch=batch,
            pad_slot_id=_PAD_SLOT_ID,
            delta_softplus_int8=delta_softplus_int8,
            output_tt=output_tt,
            ssm_states_tt=ssm_states_tt,
            z_tt=z_tt,
            u_tt=u_tt,
            delta_tt=delta_tt,
            A_tt=A_tt,
            B_tt=B_tt,
            C_tt=C_tt,
            D_tt=D_tt,
            delta_bias_tt=delta_bias_tt,
            query_start_loc_tt=query_start_loc_tt,
            cache_indices_tt=cache_indices_tt,
            has_initial_state_tt=has_initial_state_tt,
            strides=strides,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        args.dispatch_for_d_state[target](d_state)


@compiler.register_shape_function("varlen_selective_scan_fwd")
def varlen_selective_scan_fwd_shape[
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


@compiler.register("varlen_selective_state_update")
struct VarlenSelectiveStateUpdate[dt_softplus: Bool = False]:
    """Varlen selective state update for autoregressive inference.

    Performs a single step of the SSM recurrence for incremental token
    generation with multi-head support.

    Parameters:
        dt_softplus: If True, applies softplus activation to dt values.

    Tensor Shapes:
        - state: (batch, nheads, dim, d_state) - SSM state (in/out)
        - output: (batch, nheads, dim) - Output tensor
        - x: (batch, nheads, dim) - Input tensor
        - dt: (batch, nheads, dim) - Time delta tensor
        - A: (nheads, dim, d_state) - State transition matrix
        - B: (batch, ngroups, d_state) - Input matrix
        - C: (batch, ngroups, d_state) - Output matrix
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
        ctx: DeviceContext,
    ) capturing raises:
        var batch = x.dim_size(0)
        var nheads = x.dim_size(1)
        var dim = x.dim_size(2)
        var d_state = state.dim_size(3)
        var ngroups = B.dim_size(1)
        var nheads_ngroups_ratio = nheads // ngroups

        var state_tt = state.to_tile_tensor()
        var output_tt = output.to_tile_tensor()
        var x_tt = x.to_tile_tensor()
        var dt_tt = dt.to_tile_tensor()
        var A_tt = A.to_tile_tensor()
        var B_tt = B.to_tile_tensor()
        var C_tt = C.to_tile_tensor()
        var D_tt = D.to_tile_tensor()
        var z_tt = z.to_tile_tensor()
        var dt_bias_tt = dt_bias.to_tile_tensor()
        var state_batch_indices_tt = state_batch_indices.to_tile_tensor()

        var strides = VarlenSelectiveStateUpdateStrides(
            state=Strides4D(
                state.strides()[0],
                state.strides()[1],
                state.strides()[2],
                state.strides()[3],
            ),
            x=Strides3D(x.strides()[0], x.strides()[1], x.strides()[2]),
            dt=Strides3D(dt.strides()[0], dt.strides()[1], dt.strides()[2]),
            dt_bias=Strides2D(
                dt_bias.strides()[0] if dt_bias.dim_size(0) > 0 else 1,
                dt_bias.strides()[1] if dt_bias.dim_size(0) > 0 else 1,
            ),
            A=Strides3D(A.strides()[0], A.strides()[1], A.strides()[2]),
            B=Strides3D(B.strides()[0], B.strides()[1], B.strides()[2]),
            C=Strides3D(C.strides()[0], C.strides()[1], C.strides()[2]),
            D=Strides2D(
                D.strides()[0] if D.dim_size(0) > 0 else 1,
                D.strides()[1] if D.dim_size(0) > 0 else 1,
            ),
            z=Strides3D(
                z.strides()[0] if z.dim_size(0) > 0 else 1,
                z.strides()[1] if z.dim_size(0) > 0 else 1,
                z.strides()[2] if z.dim_size(0) > 0 else 1,
            ),
            output=Strides3D(
                output.strides()[0], output.strides()[1], output.strides()[2]
            ),
        )

        var has_state_batch_indices = state_batch_indices.dim_size(0) > 0
        comptime dt_softplus_int8: Int8 = Int8(1) if Self.dt_softplus else Int8(
            0
        )

        _validate_d_state(d_state)

        var total_threads: Int
        var grid_dim: Dim
        var block_dim: Dim
        comptime if is_gpu[target]():
            total_threads = (
                batch * nheads * ceildiv(dim, _GPU_UPDATE_BLOCK_SIZE_M)
            )
            grid_dim = (
                ceildiv(dim, _GPU_UPDATE_BLOCK_SIZE_M),
                batch,
                nheads,
            )
            block_dim = (1,)
        else:
            total_threads = 0
            grid_dim = (1, 1, 1)
            block_dim = (1,)

        var args = VarlenSelectiveStateUpdateArgs[
            dtype,
            state_tt.LayoutType,
            output_tt.LayoutType,
            x_tt.LayoutType,
            dt_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
            D_tt.LayoutType,
            z_tt.LayoutType,
            dt_bias_tt.LayoutType,
            state_batch_indices_tt.LayoutType,
        ](
            ctx=ctx,
            total_threads=total_threads,
            batch=batch,
            nheads=nheads,
            dim=dim,
            nheads_ngroups_ratio=nheads_ngroups_ratio,
            pad_slot_id=_PAD_SLOT_ID,
            dt_softplus_int8=dt_softplus_int8,
            has_state_batch_indices_int8=Int8(has_state_batch_indices),
            state_tt=state_tt,
            output_tt=output_tt,
            x_tt=x_tt,
            dt_tt=dt_tt,
            A_tt=A_tt,
            B_tt=B_tt,
            C_tt=C_tt,
            D_tt=D_tt,
            z_tt=z_tt,
            dt_bias_tt=dt_bias_tt,
            state_batch_indices_tt=state_batch_indices_tt,
            strides=strides,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        args.dispatch_for_d_state[target](d_state)


@compiler.register_shape_function("varlen_selective_state_update")
def varlen_selective_state_update_shape[
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
    var d_state = A.dim_size(2)
    return (IndexList[4](batch, nheads, dim, d_state), x.shape())
