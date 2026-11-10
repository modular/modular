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

import extensibility
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu

from extensibility import Input, Output, Tensor, TileTensorable
from std.utils.coord import coord_to_index_list
from std.utils.index import IndexList

from state_space.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_fwd_cpu_minimal,
    selective_scan_fwd_gpu_minimal,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
)


@extensibility.register("selective_scan_fwd")
struct SelectiveScanFwd[delta_softplus: Bool = False]:
    """Selective scan forward pass operation for Mamba SSM.

    Performs the selective scan computation used in Mamba state space models.
    This is the core operation that processes sequences through the SSM.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - out_z: (batch, dim, seqlen) - Gated output (if z is provided)
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
    """

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Output & Tensor],
        out_z: Some[Output & Tensor],
        u: Some[Input & Tensor],
        delta: Some[Input & Tensor],
        A: Some[Input & Tensor],
        B: Some[Input & Tensor],
        C: Some[Input & Tensor],
        D: Some[Input & Tensor],
        z: Some[Input & Tensor],
        delta_bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(out_z), TileTensorable)
        comptime assert conforms_to(type_of(u), TileTensorable)
        comptime assert conforms_to(type_of(delta), TileTensorable)
        comptime assert conforms_to(type_of(A), TileTensorable)
        comptime assert conforms_to(type_of(B), TileTensorable)
        comptime assert conforms_to(type_of(C), TileTensorable)
        comptime assert conforms_to(type_of(D), TileTensorable)
        comptime assert conforms_to(type_of(z), TileTensorable)
        comptime assert conforms_to(type_of(delta_bias), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(x).rank == 4
        comptime assert type_of(out_z).rank == 3
        comptime assert type_of(u).rank == 3
        comptime assert type_of(delta).rank == 3
        comptime assert type_of(A).rank == 2
        comptime assert type_of(B).rank == 4
        comptime assert type_of(C).rank == 4
        comptime assert type_of(D).rank == 1
        comptime assert type_of(z).rank == 3
        comptime assert type_of(delta_bias).rank == 1

        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = Int(output.shape()[0].value())
        var dim = Int(output.shape()[1].value())
        var seqlen = Int(output.shape()[2].value())
        var dstate = Int(A.shape()[1].value())
        var n_groups = Int(B.shape()[1].value())
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor[dtype]()
        var x_tt = x.to_tile_tensor[dtype]()
        var out_z_tt = out_z.to_tile_tensor[dtype]()
        var u_tt = u.to_tile_tensor[dtype]()
        var delta_tt = delta.to_tile_tensor[dtype]()
        var A_tt = A.to_tile_tensor[dtype]()
        var B_tt = B.to_tile_tensor[dtype]()
        var C_tt = C.to_tile_tensor[dtype]()
        var D_tt = D.to_tile_tensor[dtype]()
        var z_tt = z.to_tile_tensor[dtype]()
        var delta_bias_tt = delta_bias.to_tile_tensor[dtype]()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                String(t"Unsupported dstate: {dstate}. Expected 8 or 16.")
            )

        # Dispatch runtime dstate to compile-time DSTATE for comptime for
        # loop unrolling and guaranteed register allocation on GPU.
        comptime if ctx.target.is_cpu():
            if dstate == 16:
                selective_scan_fwd_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    Optional(ctx.assert_cpu_target()),
                )
            else:
                selective_scan_fwd_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    Optional(ctx.assert_cpu_target()),
                )
        elif ctx.target.is_gpu():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
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
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
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
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target")


@extensibility.register_shape_function("selective_scan_fwd")
def selective_scan_fwd_shape(
    u: Some[Tensor],
    delta: Some[Tensor],
    A: Some[Tensor],
    B: Some[Tensor],
    C: Some[Tensor],
    D: Some[Tensor],
    z: Some[Tensor],
    delta_bias: Some[Tensor],
) -> IndexList[3]:
    """Returns the output shape for the `selective_scan_fwd` op.

    The output of a selective scan forward pass has the same shape as the
    input sequence `u`: `(batch, dim, seqlen)`.

    Args:
        u: Input sequence tensor with shape `(batch, dim, seqlen)`.
        delta: Time-step tensor with shape `(batch, dim, seqlen)`.
        A: State transition matrix with shape `(dim, dstate)`.
        B: Input projection with shape `(batch, n_groups, dstate, seqlen)`.
        C: Output projection with shape `(batch, n_groups, dstate, seqlen)`.
        D: Skip connection with shape `(dim,)`.
        z: Gating tensor with shape `(batch, dim, seqlen)`.
        delta_bias: Delta bias with shape `(dim,)`.

    Returns:
        The shape of the output tensor `(batch, dim, seqlen)`.
    """
    comptime assert type_of(u).rank == 3, "u must be rank 3"
    comptime assert type_of(delta).rank == 3, "delta must be rank 3"
    comptime assert (
        type_of(delta).dtype == type_of(u).dtype
    ), "delta dtype must match u dtype"
    comptime assert type_of(A).rank == 2, "A must be rank 2"
    comptime assert (
        type_of(A).dtype == type_of(u).dtype
    ), "A dtype must match u dtype"
    comptime assert type_of(B).rank == 4, "B must be rank 4"
    comptime assert (
        type_of(B).dtype == type_of(u).dtype
    ), "B dtype must match u dtype"
    comptime assert type_of(C).rank == 4, "C must be rank 4"
    comptime assert (
        type_of(C).dtype == type_of(u).dtype
    ), "C dtype must match u dtype"
    comptime assert type_of(D).rank == 1, "D must be rank 1"
    comptime assert (
        type_of(D).dtype == type_of(u).dtype
    ), "D dtype must match u dtype"
    comptime assert type_of(z).rank == 3, "z must be rank 3"
    comptime assert (
        type_of(z).dtype == type_of(u).dtype
    ), "z dtype must match u dtype"
    comptime assert type_of(delta_bias).rank == 1, "delta_bias must be rank 1"
    comptime assert (
        type_of(delta_bias).dtype == type_of(u).dtype
    ), "delta_bias dtype must match u dtype"
    return rebind[IndexList[3]](coord_to_index_list(u.shape()))


@extensibility.register("selective_scan_fwd_minimal")
struct SelectiveScanFwdMinimal[delta_softplus: Bool = False]:
    """Minimal selective scan forward pass - no optional D, z, or delta_bias.

    This variant avoids passing empty tensors that could have null pointers.
    Use when D, z, and delta_bias are not provided.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
    """

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Output & Tensor],
        u: Some[Input & Tensor],
        delta: Some[Input & Tensor],
        A: Some[Input & Tensor],
        B: Some[Input & Tensor],
        C: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(u), TileTensorable)
        comptime assert conforms_to(type_of(delta), TileTensorable)
        comptime assert conforms_to(type_of(A), TileTensorable)
        comptime assert conforms_to(type_of(B), TileTensorable)
        comptime assert conforms_to(type_of(C), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(x).rank == 4
        comptime assert type_of(u).rank == 3
        comptime assert type_of(delta).rank == 3
        comptime assert type_of(A).rank == 2
        comptime assert type_of(B).rank == 4
        comptime assert type_of(C).rank == 4

        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = Int(output.shape()[0].value())
        var dim = Int(output.shape()[1].value())
        var seqlen = Int(output.shape()[2].value())
        var dstate = Int(A.shape()[1].value())
        var n_groups = Int(B.shape()[1].value())
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor[dtype]()
        var x_tt = x.to_tile_tensor[dtype]()
        var u_tt = u.to_tile_tensor[dtype]()
        var delta_tt = delta.to_tile_tensor[dtype]()
        var A_tt = A.to_tile_tensor[dtype]()
        var B_tt = B.to_tile_tensor[dtype]()
        var C_tt = C.to_tile_tensor[dtype]()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                String(t"Unsupported dstate: {dstate}. Expected 8 or 16.")
            )

        comptime if ctx.target.is_cpu():
            if dstate == 16:
                selective_scan_fwd_cpu_minimal[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    Optional(ctx.assert_cpu_target()),
                )
            else:
                selective_scan_fwd_cpu_minimal[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    Optional(ctx.assert_cpu_target()),
                )
        elif ctx.target.is_gpu():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu_minimal[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    grid_dim=(num_blocks),
                    block_dim=(BLOCK_SIZE),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu_minimal[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    grid_dim=(num_blocks),
                    block_dim=(BLOCK_SIZE),
                )
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("selective_scan_fwd_minimal")
def selective_scan_fwd_minimal_shape(
    u: Some[Tensor],
    delta: Some[Tensor],
    A: Some[Tensor],
    B: Some[Tensor],
    C: Some[Tensor],
) -> IndexList[3]:
    """Returns the output shape for the `selective_scan_fwd_minimal` op.

    The output of the minimal selective scan forward pass has the same shape
    as the input sequence `u`: `(batch, dim, seqlen)`.

    Args:
        u: Input sequence tensor with shape `(batch, dim, seqlen)`.
        delta: Time-step tensor with shape `(batch, dim, seqlen)`.
        A: State transition matrix with shape `(dim, dstate)`.
        B: Input projection with shape `(batch, n_groups, dstate, seqlen)`.
        C: Output projection with shape `(batch, n_groups, dstate, seqlen)`.

    Returns:
        The shape of the output tensor `(batch, dim, seqlen)`.
    """
    comptime assert type_of(u).rank == 3, "u must be rank 3"
    comptime assert type_of(delta).rank == 3, "delta must be rank 3"
    comptime assert (
        type_of(delta).dtype == type_of(u).dtype
    ), "delta dtype must match u dtype"
    comptime assert type_of(A).rank == 2, "A must be rank 2"
    comptime assert (
        type_of(A).dtype == type_of(u).dtype
    ), "A dtype must match u dtype"
    comptime assert type_of(B).rank == 4, "B must be rank 4"
    comptime assert (
        type_of(B).dtype == type_of(u).dtype
    ), "B dtype must match u dtype"
    comptime assert type_of(C).rank == 4, "C must be rank 4"
    comptime assert (
        type_of(C).dtype == type_of(u).dtype
    ), "C dtype must match u dtype"
    return rebind[IndexList[3]](coord_to_index_list(u.shape()))


@extensibility.register("selective_scan_update")
struct SelectiveScanUpdate[delta_softplus: Bool = False]:
    """Selective scan update operation for autoregressive inference.

    Performs a single step of the SSM recurrence for incremental token generation.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - state_out: (batch, dim, dstate) - Updated state output
        - output: (batch, dim) - Output tensor
        - state_in: (batch, dim, dstate) - Input state
        - x: (batch, dim) - Input tensor
        - dt: (batch, dim) - Time delta tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate) - Input matrix
        - C: (batch, n_groups, dstate) - Output matrix
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim) - Gating tensor (optional, can be empty)
        - dt_bias: (dim,) - Time delta bias (optional, can be empty)
    """

    @staticmethod
    def execute(
        state_out: Some[Output & Tensor],
        output: Some[Output & Tensor],
        state_in: Some[Input & Tensor],
        x: Some[Input & Tensor],
        dt: Some[Input & Tensor],
        A: Some[Input & Tensor],
        B: Some[Input & Tensor],
        C: Some[Input & Tensor],
        D: Some[Input & Tensor],
        z: Some[Input & Tensor],
        dt_bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(state_out), TileTensorable)
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(state_in), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(dt), TileTensorable)
        comptime assert conforms_to(type_of(A), TileTensorable)
        comptime assert conforms_to(type_of(B), TileTensorable)
        comptime assert conforms_to(type_of(C), TileTensorable)
        comptime assert conforms_to(type_of(D), TileTensorable)
        comptime assert conforms_to(type_of(z), TileTensorable)
        comptime assert conforms_to(type_of(dt_bias), TileTensorable)
        comptime dtype = type_of(state_out).dtype
        comptime assert type_of(state_out).rank == 3
        comptime assert type_of(output).rank == 2
        comptime assert type_of(state_in).rank == 3
        comptime assert type_of(x).rank == 2
        comptime assert type_of(dt).rank == 2
        comptime assert type_of(A).rank == 2
        comptime assert type_of(B).rank == 3
        comptime assert type_of(C).rank == 3
        comptime assert type_of(D).rank == 1
        comptime assert type_of(z).rank == 2
        comptime assert type_of(dt_bias).rank == 1

        var batch = Int(state_out.shape()[0].value())
        var dim = Int(state_out.shape()[1].value())
        var dstate = Int(state_out.shape()[2].value())
        var n_groups = Int(B.shape()[1].value())
        var group_size = dim // n_groups

        var state_out_tt = state_out.to_tile_tensor[dtype]()
        var output_tt = output.to_tile_tensor[dtype]()
        var state_in_tt = state_in.to_tile_tensor[dtype]()
        var x_tt = x.to_tile_tensor[dtype]()
        var dt_tt = dt.to_tile_tensor[dtype]()
        var A_tt = A.to_tile_tensor[dtype]()
        var B_tt = B.to_tile_tensor[dtype]()
        var C_tt = C.to_tile_tensor[dtype]()
        var D_tt = D.to_tile_tensor[dtype]()
        var z_tt = z.to_tile_tensor[dtype]()
        var dt_bias_tt = dt_bias.to_tile_tensor[dtype]()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                String(t"Unsupported dstate: {dstate}. Expected 8 or 16.")
            )

        comptime if ctx.target.is_cpu():
            if dstate == 16:
                selective_scan_update_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    Optional(ctx.assert_cpu_target()),
                )
            else:
                selective_scan_update_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    Optional(ctx.assert_cpu_target()),
                )
        elif ctx.target.is_gpu():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
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
                        state_out_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(group_size),
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
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
                        state_out_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(group_size),
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target")


@extensibility.register_shape_function("selective_scan_update")
def selective_scan_update_shape(
    state_in: Some[Tensor],
    x: Some[Tensor],
    dt: Some[Tensor],
    A: Some[Tensor],
    B: Some[Tensor],
    C: Some[Tensor],
    D: Some[Tensor],
    z: Some[Tensor],
    dt_bias: Some[Tensor],
) -> Tuple[IndexList[3], IndexList[2]]:
    """Returns the output shapes for the `selective_scan_update` op.

    The update step produces two tensors: the updated SSM state and the
    single-step output for the new token.

    Args:
        state_in: Input SSM state with shape `(batch, dim, dstate)`.
        x: Input token with shape `(batch, dim)`.
        dt: Time-delta tensor with shape `(batch, dim)`.
        A: State transition matrix with shape `(dim, dstate)`.
        B: Input matrix with shape `(batch, n_groups, dstate)`.
        C: Output matrix with shape `(batch, n_groups, dstate)`.
        D: Skip connection with shape `(dim,)`.
        z: Gating tensor with shape `(batch, dim)`.
        dt_bias: Time-delta bias with shape `(dim,)`.

    Returns:
        A tuple of `(state_out_shape, output_shape)` where `state_out_shape`
        matches `state_in.shape()` and `output_shape` matches `x.shape()`.
    """
    comptime assert type_of(state_in).rank == 3, "state_in must be rank 3"
    comptime assert type_of(x).rank == 2, "x must be rank 2"
    comptime assert (
        type_of(x).dtype == type_of(state_in).dtype
    ), "x dtype must match state_in dtype"
    comptime assert type_of(dt).rank == 2, "dt must be rank 2"
    comptime assert (
        type_of(dt).dtype == type_of(state_in).dtype
    ), "dt dtype must match state_in dtype"
    comptime assert type_of(A).rank == 2, "A must be rank 2"
    comptime assert (
        type_of(A).dtype == type_of(state_in).dtype
    ), "A dtype must match state_in dtype"
    comptime assert type_of(B).rank == 3, "B must be rank 3"
    comptime assert (
        type_of(B).dtype == type_of(state_in).dtype
    ), "B dtype must match state_in dtype"
    comptime assert type_of(C).rank == 3, "C must be rank 3"
    comptime assert (
        type_of(C).dtype == type_of(state_in).dtype
    ), "C dtype must match state_in dtype"
    comptime assert type_of(D).rank == 1, "D must be rank 1"
    comptime assert (
        type_of(D).dtype == type_of(state_in).dtype
    ), "D dtype must match state_in dtype"
    comptime assert type_of(z).rank == 2, "z must be rank 2"
    comptime assert (
        type_of(z).dtype == type_of(state_in).dtype
    ), "z dtype must match state_in dtype"
    comptime assert type_of(dt_bias).rank == 1, "dt_bias must be rank 1"
    comptime assert (
        type_of(dt_bias).dtype == type_of(state_in).dtype
    ), "dt_bias dtype must match state_in dtype"
    return (
        rebind[IndexList[3]](coord_to_index_list(state_in.shape())),
        rebind[IndexList[2]](coord_to_index_list(x.shape())),
    )
