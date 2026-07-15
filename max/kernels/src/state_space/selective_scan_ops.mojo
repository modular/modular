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
"""Selective scan operation registrations for Mamba SSM.

This module registers the following ops:
- selective_scan_fwd: Full selective scan forward pass
- selective_scan_fwd_minimal: Minimal variant without optional tensors
- selective_scan_update: Single-step update for autoregressive inference
"""

from std.math import ceildiv

import extensibility as compiler
from std.gpu.host import DeviceContext, Dim
from std.gpu.host.info import is_cpu, is_gpu

from extensibility import InputTensor, OutputTensor
from layout import TensorLayout, TileTensor
from std.utils.index import IndexList

from state_space.selective_scan import (
    Strides1D,
    Strides2D,
    Strides3D,
    Strides4D,
    selective_scan_fwd_cpu,
    selective_scan_fwd_cpu_minimal,
    selective_scan_fwd_gpu,
    selective_scan_fwd_gpu_minimal,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
)

comptime _GPU_BLOCK_SIZE = 128


@fieldwise_init
struct SelectiveScanFwdStrides(Copyable, ImplicitlyCopyable, Movable):
    var output: Strides3D
    var x: Strides4D
    var out_z: Strides3D
    var u: Strides3D
    var delta: Strides3D
    var A: Strides2D
    var B: Strides4D
    var C: Strides4D
    var D: Strides1D
    var z: Strides3D
    var delta_bias: Strides1D


@fieldwise_init
struct SelectiveScanFwdMinimalStrides(Copyable, ImplicitlyCopyable, Movable):
    var output: Strides3D
    var x: Strides4D
    var u: Strides3D
    var delta: Strides3D
    var A: Strides2D
    var B: Strides4D
    var C: Strides4D


@fieldwise_init
struct SelectiveScanUpdateStrides(Copyable, ImplicitlyCopyable, Movable):
    var state_out: Strides3D
    var output: Strides2D
    var state_in: Strides3D
    var x: Strides2D
    var dt: Strides2D
    var A: Strides2D
    var B: Strides3D
    var C: Strides3D
    var D: Strides1D
    var z: Strides2D
    var dt_bias: Strides1D


# Supported d_state values for the non-varlen kernels, largest first. This is
# the single source of truth: _validate_d_state and every dispatch_for_d_state
# below iterate it. Keep _SUPPORTED_D_STATE_VALUES (the human-readable error
# text) in sync with it.
comptime _SUPPORTED_D_STATE = [256, 128, 64, 32, 16, 8]
comptime _SUPPORTED_D_STATE_VALUES = "8, 16, 32, 64, 128, or 256"


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
struct SelectiveScanFwdArgs[
    dtype: DType,
    output_LT: TensorLayout,
    x_LT: TensorLayout,
    out_z_LT: TensorLayout,
    u_LT: TensorLayout,
    delta_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    z_LT: TensorLayout,
    delta_bias_LT: TensorLayout,
]:
    var ctx: DeviceContext
    var total_batch_dim: Int
    var batch: Int
    var dim: Int
    var seqlen: Int
    var group_size: Int
    var delta_softplus_int8: Int8
    var output_tt: TileTensor[
        mut=True, Self.dtype, Self.output_LT, MutUntrackedOrigin
    ]
    var x_tt: TileTensor[mut=True, Self.dtype, Self.x_LT, MutUntrackedOrigin]
    var out_z_tt: TileTensor[
        mut=True, Self.dtype, Self.out_z_LT, MutUntrackedOrigin
    ]
    var u_tt: TileTensor[Self.dtype, Self.u_LT, MutUntrackedOrigin]
    var delta_tt: TileTensor[Self.dtype, Self.delta_LT, MutUntrackedOrigin]
    var A_tt: TileTensor[Self.dtype, Self.A_LT, MutUntrackedOrigin]
    var B_tt: TileTensor[Self.dtype, Self.B_LT, MutUntrackedOrigin]
    var C_tt: TileTensor[Self.dtype, Self.C_LT, MutUntrackedOrigin]
    var D_tt: TileTensor[Self.dtype, Self.D_LT, MutUntrackedOrigin]
    var z_tt: TileTensor[Self.dtype, Self.z_LT, MutUntrackedOrigin]
    var delta_bias_tt: TileTensor[
        Self.dtype, Self.delta_bias_LT, MutUntrackedOrigin
    ]
    var strides: SelectiveScanFwdStrides
    var grid_dim: Dim
    var block_dim: Dim

    @parameter
    def launch_gpu[d_state_val: Int](self) capturing raises:
        comptime kernel = selective_scan_fwd_gpu[
            Self.dtype,
            d_state_val,
            Self.output_LT,
            Self.x_LT,
            Self.out_z_LT,
            Self.u_LT,
            Self.delta_LT,
            Self.A_LT,
            Self.B_LT,
            Self.C_LT,
            Self.D_LT,
            Self.z_LT,
            Self.delta_bias_LT,
        ]
        var compiled_kernel = self.ctx.compile_function[kernel]()
        self.ctx.enqueue_function(
            compiled_kernel,
            self.total_batch_dim,
            self.batch,
            self.dim,
            self.seqlen,
            self.group_size,
            self.delta_softplus_int8,
            self.output_tt,
            self.x_tt,
            self.out_z_tt,
            self.u_tt,
            self.delta_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.delta_bias_tt,
            self.strides.output,
            self.strides.x,
            self.strides.out_z,
            self.strides.u,
            self.strides.delta,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.delta_bias,
            grid_dim=self.grid_dim,
            block_dim=self.block_dim,
        )

    @parameter
    def run_cpu[d_state_val: Int](self) capturing raises:
        selective_scan_fwd_cpu[
            Self.dtype,
            d_state_val,
        ](
            self.batch,
            self.dim,
            self.seqlen,
            self.group_size,
            self.delta_softplus_int8,
            self.output_tt,
            self.x_tt,
            self.out_z_tt,
            self.u_tt,
            self.delta_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.delta_bias_tt,
            self.strides.output,
            self.strides.x,
            self.strides.out_z,
            self.strides.u,
            self.strides.delta,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.delta_bias,
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
struct SelectiveScanFwdMinimalArgs[
    dtype: DType,
    output_LT: TensorLayout,
    x_LT: TensorLayout,
    u_LT: TensorLayout,
    delta_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
]:
    var ctx: DeviceContext
    var total_batch_dim: Int
    var batch: Int
    var dim: Int
    var seqlen: Int
    var group_size: Int
    var delta_softplus_int8: Int8
    var output_tt: TileTensor[
        mut=True, Self.dtype, Self.output_LT, MutUntrackedOrigin
    ]
    var x_tt: TileTensor[mut=True, Self.dtype, Self.x_LT, MutUntrackedOrigin]
    var u_tt: TileTensor[Self.dtype, Self.u_LT, MutUntrackedOrigin]
    var delta_tt: TileTensor[Self.dtype, Self.delta_LT, MutUntrackedOrigin]
    var A_tt: TileTensor[Self.dtype, Self.A_LT, MutUntrackedOrigin]
    var B_tt: TileTensor[Self.dtype, Self.B_LT, MutUntrackedOrigin]
    var C_tt: TileTensor[Self.dtype, Self.C_LT, MutUntrackedOrigin]
    var strides: SelectiveScanFwdMinimalStrides
    var grid_dim: Dim
    var block_dim: Dim

    @parameter
    def launch_gpu[d_state_val: Int](self) capturing raises:
        comptime kernel = selective_scan_fwd_gpu_minimal[
            Self.dtype,
            d_state_val,
            Self.output_LT,
            Self.x_LT,
            Self.u_LT,
            Self.delta_LT,
            Self.A_LT,
            Self.B_LT,
            Self.C_LT,
        ]
        var compiled_kernel = self.ctx.compile_function[kernel]()
        self.ctx.enqueue_function(
            compiled_kernel,
            self.total_batch_dim,
            self.batch,
            self.dim,
            self.seqlen,
            self.group_size,
            self.delta_softplus_int8,
            self.output_tt,
            self.x_tt,
            self.u_tt,
            self.delta_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.strides.output,
            self.strides.x,
            self.strides.u,
            self.strides.delta,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            grid_dim=self.grid_dim,
            block_dim=self.block_dim,
        )

    @parameter
    def run_cpu[d_state_val: Int](self) capturing raises:
        selective_scan_fwd_cpu_minimal[
            Self.dtype,
            d_state_val,
        ](
            self.batch,
            self.dim,
            self.seqlen,
            self.group_size,
            self.delta_softplus_int8,
            self.output_tt,
            self.x_tt,
            self.u_tt,
            self.delta_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.strides.output,
            self.strides.x,
            self.strides.u,
            self.strides.delta,
            self.strides.A,
            self.strides.B,
            self.strides.C,
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
struct SelectiveScanUpdateArgs[
    dtype: DType,
    state_out_LT: TensorLayout,
    output_LT: TensorLayout,
    state_in_LT: TensorLayout,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    z_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
]:
    var ctx: DeviceContext
    var total_batch_dim: Int
    var batch: Int
    var dim: Int
    var group_size: Int
    var delta_softplus_int8: Int8
    var state_out_tt: TileTensor[
        mut=True, Self.dtype, Self.state_out_LT, MutUntrackedOrigin
    ]
    var output_tt: TileTensor[
        mut=True, Self.dtype, Self.output_LT, MutUntrackedOrigin
    ]
    var state_in_tt: TileTensor[
        Self.dtype, Self.state_in_LT, MutUntrackedOrigin
    ]
    var x_tt: TileTensor[Self.dtype, Self.x_LT, MutUntrackedOrigin]
    var dt_tt: TileTensor[Self.dtype, Self.dt_LT, MutUntrackedOrigin]
    var A_tt: TileTensor[Self.dtype, Self.A_LT, MutUntrackedOrigin]
    var B_tt: TileTensor[Self.dtype, Self.B_LT, MutUntrackedOrigin]
    var C_tt: TileTensor[Self.dtype, Self.C_LT, MutUntrackedOrigin]
    var D_tt: TileTensor[Self.dtype, Self.D_LT, MutUntrackedOrigin]
    var z_tt: TileTensor[Self.dtype, Self.z_LT, MutUntrackedOrigin]
    var dt_bias_tt: TileTensor[Self.dtype, Self.dt_bias_LT, MutUntrackedOrigin]
    var strides: SelectiveScanUpdateStrides
    var grid_dim: Dim
    var block_dim: Dim

    @parameter
    def launch_gpu[d_state_val: Int](self) capturing raises:
        comptime kernel = selective_scan_update_gpu[
            Self.dtype,
            d_state_val,
            Self.state_out_LT,
            Self.output_LT,
            Self.state_in_LT,
            Self.x_LT,
            Self.dt_LT,
            Self.A_LT,
            Self.B_LT,
            Self.C_LT,
            Self.D_LT,
            Self.z_LT,
            Self.dt_bias_LT,
        ]
        var compiled_kernel = self.ctx.compile_function[kernel]()
        self.ctx.enqueue_function(
            compiled_kernel,
            self.total_batch_dim,
            self.batch,
            self.dim,
            self.group_size,
            self.delta_softplus_int8,
            self.state_out_tt,
            self.output_tt,
            self.state_in_tt,
            self.x_tt,
            self.dt_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.dt_bias_tt,
            self.strides.state_out,
            self.strides.output,
            self.strides.state_in,
            self.strides.x,
            self.strides.dt,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.dt_bias,
            grid_dim=self.grid_dim,
            block_dim=self.block_dim,
        )

    @parameter
    def run_cpu[d_state_val: Int](self) capturing raises:
        selective_scan_update_cpu[
            Self.dtype,
            d_state_val,
        ](
            self.batch,
            self.dim,
            self.group_size,
            self.delta_softplus_int8,
            self.state_out_tt,
            self.output_tt,
            self.state_in_tt,
            self.x_tt,
            self.dt_tt,
            self.A_tt,
            self.B_tt,
            self.C_tt,
            self.D_tt,
            self.z_tt,
            self.dt_bias_tt,
            self.strides.state_out,
            self.strides.output,
            self.strides.state_in,
            self.strides.x,
            self.strides.dt,
            self.strides.A,
            self.strides.B,
            self.strides.C,
            self.strides.D,
            self.strides.z,
            self.strides.dt_bias,
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


@compiler.register("selective_scan_fwd")
struct SelectiveScanFwd[delta_softplus: Bool = False]:
    """Selective scan forward pass operation for Mamba SSM.

    Performs the selective scan computation used in Mamba state space models.
    This is the core operation that processes sequences through the SSM.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*d_state) - Checkpoint tensor for chunking
        - out_z: (batch, dim, seqlen) - Gated output (if z is provided)
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, d_state) - State transition matrix
        - B: (batch, n_groups, d_state, seqlen) - Input projection
        - C: (batch, n_groups, d_state, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        out_z: OutputTensor[dtype=dtype, rank=3, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var d_state = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor()
        var x_tt = x.to_tile_tensor()
        var out_z_tt = out_z.to_tile_tensor()
        var u_tt = u.to_tile_tensor()
        var delta_tt = delta.to_tile_tensor()
        var A_tt = A.to_tile_tensor()
        var B_tt = B.to_tile_tensor()
        var C_tt = C.to_tile_tensor()
        var D_tt = D.to_tile_tensor()
        var z_tt = z.to_tile_tensor()
        var delta_bias_tt = delta_bias.to_tile_tensor()

        var strides = SelectiveScanFwdStrides(
            output=output.strides(),
            x=x.strides(),
            out_z=out_z.strides(),
            u=u.strides(),
            delta=delta.strides(),
            A=A.strides(),
            B=B.strides(),
            C=C.strides(),
            D=D.strides(),
            z=z.strides(),
            delta_bias=delta_bias.strides(),
        )

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        _validate_d_state(d_state)

        var total_batch_dim: Int
        var grid_dim: Dim
        var block_dim: Dim
        comptime if is_gpu[target]():
            var num_blocks = ceildiv(batch * dim, _GPU_BLOCK_SIZE)
            total_batch_dim = batch * dim
            grid_dim = (num_blocks,)
            block_dim = (_GPU_BLOCK_SIZE,)
        else:
            total_batch_dim = 0
            grid_dim = (1,)
            block_dim = (1,)

        var args = SelectiveScanFwdArgs[
            dtype,
            output_tt.LayoutType,
            x_tt.LayoutType,
            out_z_tt.LayoutType,
            u_tt.LayoutType,
            delta_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
            D_tt.LayoutType,
            z_tt.LayoutType,
            delta_bias_tt.LayoutType,
        ](
            ctx=ctx,
            total_batch_dim=total_batch_dim,
            batch=batch,
            dim=dim,
            seqlen=seqlen,
            group_size=group_size,
            delta_softplus_int8=delta_softplus_int8,
            output_tt=output_tt,
            x_tt=x_tt,
            out_z_tt=out_z_tt,
            u_tt=u_tt,
            delta_tt=delta_tt,
            A_tt=A_tt,
            B_tt=B_tt,
            C_tt=C_tt,
            D_tt=D_tt,
            z_tt=z_tt,
            delta_bias_tt=delta_bias_tt,
            strides=strides,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        args.dispatch_for_d_state[target](d_state)


@compiler.register_shape_function("selective_scan_fwd")
def selective_scan_fwd_shape[
    dtype: DType,
](
    u: InputTensor[dtype=dtype, rank=3, ...],
    delta: InputTensor[dtype=dtype, rank=3, ...],
    A: InputTensor[dtype=dtype, rank=2, ...],
    B: InputTensor[dtype=dtype, rank=4, ...],
    C: InputTensor[dtype=dtype, rank=4, ...],
    D: InputTensor[dtype=dtype, rank=1, ...],
    z: InputTensor[dtype=dtype, rank=3, ...],
    delta_bias: InputTensor[dtype=dtype, rank=1, ...],
) -> IndexList[3]:
    return u.shape()


@compiler.register("selective_scan_fwd_minimal")
struct SelectiveScanFwdMinimal[delta_softplus: Bool = False]:
    """Minimal selective scan forward pass - no optional D, z, or delta_bias.

    This variant avoids passing empty tensors that could have null pointers.
    Use when D, z, and delta_bias are not provided.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*d_state) - Checkpoint tensor for chunking
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, d_state) - State transition matrix
        - B: (batch, n_groups, d_state, seqlen) - Input projection
        - C: (batch, n_groups, d_state, seqlen) - Output projection
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var d_state = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor()
        var x_tt = x.to_tile_tensor()
        var u_tt = u.to_tile_tensor()
        var delta_tt = delta.to_tile_tensor()
        var A_tt = A.to_tile_tensor()
        var B_tt = B.to_tile_tensor()
        var C_tt = C.to_tile_tensor()

        var strides = SelectiveScanFwdMinimalStrides(
            output=output.strides(),
            x=x.strides(),
            u=u.strides(),
            delta=delta.strides(),
            A=A.strides(),
            B=B.strides(),
            C=C.strides(),
        )

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        _validate_d_state(d_state)

        var total_batch_dim: Int
        var grid_dim: Dim
        var block_dim: Dim
        comptime if is_gpu[target]():
            var num_blocks = ceildiv(batch * dim, _GPU_BLOCK_SIZE)
            total_batch_dim = batch * dim
            grid_dim = (num_blocks,)
            block_dim = (_GPU_BLOCK_SIZE,)
        else:
            total_batch_dim = 0
            grid_dim = (1,)
            block_dim = (1,)

        var args = SelectiveScanFwdMinimalArgs[
            dtype,
            output_tt.LayoutType,
            x_tt.LayoutType,
            u_tt.LayoutType,
            delta_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
        ](
            ctx=ctx,
            total_batch_dim=total_batch_dim,
            batch=batch,
            dim=dim,
            seqlen=seqlen,
            group_size=group_size,
            delta_softplus_int8=delta_softplus_int8,
            output_tt=output_tt,
            x_tt=x_tt,
            u_tt=u_tt,
            delta_tt=delta_tt,
            A_tt=A_tt,
            B_tt=B_tt,
            C_tt=C_tt,
            strides=strides,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        args.dispatch_for_d_state[target](d_state)


@compiler.register_shape_function("selective_scan_fwd_minimal")
def selective_scan_fwd_minimal_shape[
    dtype: DType,
](
    u: InputTensor[dtype=dtype, rank=3, ...],
    delta: InputTensor[dtype=dtype, rank=3, ...],
    A: InputTensor[dtype=dtype, rank=2, ...],
    B: InputTensor[dtype=dtype, rank=4, ...],
    C: InputTensor[dtype=dtype, rank=4, ...],
) -> IndexList[3]:
    return u.shape()


@compiler.register("selective_scan_update")
struct SelectiveScanUpdate[delta_softplus: Bool = False]:
    """Selective scan update operation for autoregressive inference.

    Performs a single step of the SSM recurrence for incremental token generation.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - state_out: (batch, dim, d_state) - Updated state output
        - output: (batch, dim) - Output tensor
        - state_in: (batch, dim, d_state) - Input state
        - x: (batch, dim) - Input tensor
        - dt: (batch, dim) - Time delta tensor
        - A: (dim, d_state) - State transition matrix
        - B: (batch, n_groups, d_state) - Input matrix
        - C: (batch, n_groups, d_state) - Output matrix
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim) - Gating tensor (optional, can be empty)
        - dt_bias: (dim,) - Time delta bias (optional, can be empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        state_out: OutputTensor[dtype=dtype, rank=3, ...],
        output: OutputTensor[dtype=dtype, rank=2, ...],
        state_in: InputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        dt: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=2, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var batch = state_out.dim_size(0)
        var dim = state_out.dim_size(1)
        var d_state = state_out.dim_size(2)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var state_out_tt = state_out.to_tile_tensor()
        var output_tt = output.to_tile_tensor()
        var state_in_tt = state_in.to_tile_tensor()
        var x_tt = x.to_tile_tensor()
        var dt_tt = dt.to_tile_tensor()
        var A_tt = A.to_tile_tensor()
        var B_tt = B.to_tile_tensor()
        var C_tt = C.to_tile_tensor()
        var D_tt = D.to_tile_tensor()
        var z_tt = z.to_tile_tensor()
        var dt_bias_tt = dt_bias.to_tile_tensor()

        var strides = SelectiveScanUpdateStrides(
            state_out=state_out.strides(),
            output=output.strides(),
            state_in=state_in.strides(),
            x=x.strides(),
            dt=dt.strides(),
            A=A.strides(),
            B=B.strides(),
            C=C.strides(),
            D=D.strides(),
            z=z.strides(),
            dt_bias=dt_bias.strides(),
        )

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        _validate_d_state(d_state)

        var total_batch_dim: Int
        var grid_dim: Dim
        var block_dim: Dim
        comptime if is_gpu[target]():
            var num_blocks = ceildiv(batch * dim, _GPU_BLOCK_SIZE)
            total_batch_dim = batch * dim
            grid_dim = (num_blocks,)
            block_dim = (_GPU_BLOCK_SIZE,)
        else:
            total_batch_dim = 0
            grid_dim = (1,)
            block_dim = (1,)

        var args = SelectiveScanUpdateArgs[
            dtype,
            state_out_tt.LayoutType,
            output_tt.LayoutType,
            state_in_tt.LayoutType,
            x_tt.LayoutType,
            dt_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
            D_tt.LayoutType,
            z_tt.LayoutType,
            dt_bias_tt.LayoutType,
        ](
            ctx=ctx,
            total_batch_dim=total_batch_dim,
            batch=batch,
            dim=dim,
            group_size=group_size,
            delta_softplus_int8=delta_softplus_int8,
            state_out_tt=state_out_tt,
            output_tt=output_tt,
            state_in_tt=state_in_tt,
            x_tt=x_tt,
            dt_tt=dt_tt,
            A_tt=A_tt,
            B_tt=B_tt,
            C_tt=C_tt,
            D_tt=D_tt,
            z_tt=z_tt,
            dt_bias_tt=dt_bias_tt,
            strides=strides,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        args.dispatch_for_d_state[target](d_state)


@compiler.register_shape_function("selective_scan_update")
def selective_scan_update_shape[
    dtype: DType,
](
    state_in: InputTensor[dtype=dtype, rank=3, ...],
    x: InputTensor[dtype=dtype, rank=2, ...],
    dt: InputTensor[dtype=dtype, rank=2, ...],
    A: InputTensor[dtype=dtype, rank=2, ...],
    B: InputTensor[dtype=dtype, rank=3, ...],
    C: InputTensor[dtype=dtype, rank=3, ...],
    D: InputTensor[dtype=dtype, rank=1, ...],
    z: InputTensor[dtype=dtype, rank=2, ...],
    dt_bias: InputTensor[dtype=dtype, rank=1, ...],
) -> Tuple[IndexList[3], IndexList[2]]:
    return (state_in.shape(), x.shape())
