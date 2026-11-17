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

# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers the per-device (`mo.bundled.*`) collective graph ops backed by the `comm` kernels.

A bundled collective is the single-device form of a `mo.distributed.*` op:
one instance runs per GPU, reading its peers' tensors directly. `execute`
reads them through live data pointers; `build_graph` reads them through
stable address slots so a recorded device graph can be replayed.
"""

from std.collections import Array
from std.math import align_up, ceildiv
from std.sys.info import size_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from comm import Signal
from comm.allgather import allgather
from comm.allreduce import allreduce
from comm.allreduce_residual_rmsnorm import allreduce_residual_rmsnorm
from comm.broadcast import broadcast
from comm.reducescatter import CollectiveEpilogueFn
from extensibility import (
    Fused,
    Input,
    MutableInput,
    Output,
    Stable,
    Storable,
    StoreAccess,
    Tensor,
    TensorTuple,
    TileTensorable,
    UnsafeFusedView,
    unsafe_fused_view,
)
from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceContext, DeviceGraphBuilder

from .kernels import (
    _check_signal_buffer_size,
    _check_tensor_tuple,
    _partitioned_scratch_requirement,
)


@extensibility.register("mo.bundled.allreduce.sum")
struct BundledAllReduceSum:
    """Registers the `mo.bundled.allreduce.sum` graph op with the graph compiler.
    """

    @staticmethod
    def _launch[
        dtype: DType,
        num_devices: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        OutputLayout: TensorLayout,
        OutputLambda: CollectiveEpilogueFn,
        //,
        group_size: Int,
        peer_rank: Optional[Int],
    ](
        in_tensors: Array[
            TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
            num_devices,
        ],
        out_layout: OutputLayout,
        output_lambda: OutputLambda,
        rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], num_devices],
        signal_buffer_size: Int,
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            group_size >= 1 and num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # The peer arrays are indexed by global rank. Without an explicit
        # rank, the device id stands in for it, which is right only when the
        # world is devices 0..N-1 in order; under recording the context
        # reports the builder's device.
        var my_rank = peer_rank.or_else(Int(ctx.device().id))
        var group_start = (my_rank // group_size) * group_size

        # allreduce 2-stage uses size/ngpus scratch space, per group; sibling
        # groups may differ in shape, so size it from this group's input.
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            group_size, dtype
        ](in_tensors[group_start].num_elements())
        _check_signal_buffer_size(signal_buffer_size, scratch_buffer_size_bytes)

        allreduce[ngpus=num_devices, group_size=group_size](
            in_tensors,
            out_layout,
            output_lambda,
            rank_sigs,
            ctx,
            my_rank=my_rank,
        )

    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        group_size: Int = 0,
        peer_rank: Optional[Int] = None,
    ](
        output: Some[Output & Fused & StoreAccess],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        ctx: DeviceContext,
    ) raises:
        """Per-device allreduce sum, for use with mo.parallel dispatch.

        This path does not take the kernel-load gate: `mo.parallel` emits N
        independent async stubs, with no single point to claim or release it,
        so a first-time kernel load can still interleave with these launches.

        Parameters:
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name used for profiling and debugging.
            group_size: Number of contiguous devices per independent
                allreduce group; 0 means one full-world group.
            peer_rank: THIS GPU's index in the world, so in `inputs` and
                `signal_buffers`; `None` takes it from the device id.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Input tensors from ALL GPUs in the world.
            signal_buffers: Signal buffers for ALL GPUs in the world.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert type_of(output).dtype == dtype
        comptime assert type_of(output).rank == rank
        comptime effective_group_size = (
            num_devices if group_size == 0 else group_size
        )

        # A collective addresses the signal buffers through flat pointers; a
        # non-pointer storage policy aborts here at run time.
        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[dtype]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[dtype]().as_imm()
            )

        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)

        # Writes this device's reduced output into its fused epilogue output.
        # It runs on the device, so it copies the view.
        @inline(.always)
        def output_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) {var out_view} -> None:
            comptime assert conforms_to(type_of(out_view), Storable)
            out_view.store[width=_width, element_alignment=alignment](
                coords, rebind[SIMD[type_of(out_view).dtype, _width]](val)
            )

        Self._launch[group_size=effective_group_size, peer_rank=peer_rank](
            in_tensors,
            row_major(output.shape()),
            output_lambda,
            rank_sigs,
            Int(signal_buffers[0].shape().product()),
            ctx,
        )

    @staticmethod
    def build_graph[
        Ins: TypeList[Trait=Input & Tensor & Stable, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        group_size: Int = 0,
        peer_rank: Optional[Int] = None,
    ](
        output: Some[Output & Fused & StoreAccess],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device allreduce sum into a device graph.

        The peers arrive as stable address slots: the recorded kernels read
        each peer's data pointer out of its slot at replay time, so the graph
        does not bake in the peer buffers themselves.

        Parameters:
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name used for profiling and debugging.
            group_size: Number of contiguous devices per independent
                allreduce group; 0 means one full-world group.
            peer_rank: THIS GPU's index in the world, so in `inputs` and
                `signal_buffers`; `None` takes it from the device id.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Slot-backed input tensors from ALL GPUs in the world.
            signal_buffers: Signal buffers for ALL GPUs in the world.
            builder: Device graph builder for THIS GPU.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert type_of(output).dtype == dtype
        comptime assert type_of(output).rank == rank
        comptime effective_group_size = (
            num_devices if group_size == 0 else group_size
        )

        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = type_of(inputs[0].to_tile_tensor[dtype]())
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[dtype]()
            )

        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)

        @inline(.always)
        def output_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) {var out_view} -> None:
            comptime assert conforms_to(type_of(out_view), Storable)
            out_view.store[width=_width, element_alignment=alignment](
                coords, rebind[SIMD[type_of(out_view).dtype, _width]](val)
            )

        Self._launch[group_size=effective_group_size, peer_rank=peer_rank](
            in_tensors,
            row_major(output.shape()),
            output_lambda,
            rank_sigs,
            Int(signal_buffers[0].shape().product()),
            builder.recording_context(),
        )


@extensibility.register("mo.bundled.broadcast")
struct BundledBroadcast:
    """Registers the `mo.bundled.broadcast` graph op with the graph compiler.

    The per-device form of `mo.distributed.broadcast`: one instance per GPU
    copies the root device's tensor into its own output. `execute` reads the
    root tensor through its live data pointer; `build_graph` reads it through
    a stable address slot (a `Stable` input), so the recorded device graph
    replays against whatever buffer the slot names at the time.
    """

    @staticmethod
    def _launch[
        dtype: DType,
        num_devices: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        out_engine: TensorEngine,
        //,
        root: Int,
        peer_rank: Optional[Int],
    ](
        in_buf: TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
        out_buf: TileTensor[mut=True, dtype, in_layout, _, Engine=out_engine],
        input_size: Int,
        rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], num_devices],
        signal_buffer_size: Int,
        ctx: DeviceContext,
    ) raises:
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input_size)
        _check_signal_buffer_size(signal_buffer_size, scratch_buffer_size_bytes)

        # The signal buffers are indexed by global rank. Without an explicit
        # rank, the device id stands in for it, which is right only when the
        # world is devices 0..N-1 in order; under recording the context
        # reports the builder's device.
        var my_rank = peer_rank.or_else(Int(ctx.device().id))
        broadcast[num_devices](
            in_buf, out_buf, rank_sigs, ctx, root, rank=my_rank
        )

    @staticmethod
    def execute[
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        root: Int,
        _trace_name: StaticString,
        peer_rank: Optional[Int] = None,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        signal_buffers: TensorTuple[*Sigs],
        ctx: DeviceContext,
    ) raises:
        """Per-device broadcast through live addresses.

        Parameters:
            Sigs: The element types of `signal_buffers` (inferred).
            root: Index of the root device in the signal buffer list.
            _trace_name: Trace name used for profiling and debugging.
            peer_rank: THIS GPU's index in the signal buffer list; `None` takes it from the device id.

        Args:
            output: Output tensor for THIS GPU.
            input: The root GPU's tensor.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = len(Sigs)
        comptime dtype = type_of(input).dtype
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert type_of(output).dtype == dtype
        comptime assert type_of(output).rank == type_of(input).rank

        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(output), TileTensorable)
        var in_buf = input.to_tile_tensor[dtype]()
        var out_buf = TileTensor(
            ptr=output.to_tile_tensor[dtype]().unsafe_ptr(),
            layout=in_buf.layout,
        )
        Self._launch[root=root, peer_rank=peer_rank](
            in_buf,
            out_buf,
            Int(input.shape().product()),
            rank_sigs,
            Int(signal_buffers[0].shape().product()),
            ctx,
        )

    @staticmethod
    def build_graph[
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        root: Int,
        _trace_name: StaticString,
        peer_rank: Optional[Int] = None,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor & Stable],
        signal_buffers: TensorTuple[*Sigs],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device broadcast into a device graph.

        Parameters:
            Sigs: The element types of `signal_buffers` (inferred).
            root: Index of the root device in the signal buffer list.
            _trace_name: Trace name used for profiling and debugging.
            peer_rank: THIS GPU's index in the signal buffer list; `None` takes it from the device id.

        Args:
            output: Output tensor for THIS GPU.
            input: The root GPU's tensor, read through its stable slot.
            signal_buffers: Signal buffers for ALL participating GPUs.
            builder: Device graph builder for THIS GPU.
        """
        comptime num_devices = len(Sigs)
        comptime dtype = type_of(input).dtype
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert type_of(output).dtype == dtype
        comptime assert type_of(output).rank == type_of(input).rank

        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(output), TileTensorable)
        var in_buf = input.to_tile_tensor[dtype]()
        var out_buf = TileTensor(
            ptr=output.to_tile_tensor[dtype]().unsafe_ptr(),
            layout=in_buf.layout,
        )
        Self._launch[root=root, peer_rank=peer_rank](
            in_buf,
            out_buf,
            Int(input.shape().product()),
            rank_sigs,
            Int(signal_buffers[0].shape().product()),
            builder.recording_context(),
        )


@extensibility.register("mo.bundled.allgather")
struct BundledAllGather:
    """Registers the `mo.bundled.allgather` graph op with the graph compiler.

    The per-device form of `mo.distributed.allgather`, full world only: one
    instance per GPU gathers every peer's tensor into its own outputs, one
    per peer in peer order. `execute` reads the peers through live data
    pointers; `build_graph` reads them through stable address slots
    (`Stable` inputs).
    """

    @staticmethod
    def _launch[
        dtype: DType,
        num_devices: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        out_layout: TensorLayout,
        out_origin: MutOrigin,
        out_engine: TensorEngine,
        //,
        peer_rank: Optional[Int],
    ](
        in_tensors: Array[
            TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
            num_devices,
        ],
        out_tiles: Array[
            TileTensor[
                mut=True, dtype, out_layout, out_origin, Engine=out_engine
            ],
            num_devices,
        ],
        rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], num_devices],
        signal_buffer_size: Int,
        ctx: DeviceContext,
    ) raises:
        _check_signal_buffer_size(signal_buffer_size, 0)

        # The kernel takes every device's outputs, laid out
        # `[device * num_devices + source]`, but a full-world gather writes
        # only this device's slice, so the peers' slots alias it.
        var out_tensors = Array[
            type_of(out_tiles[0]), num_devices * num_devices
        ](uninitialized=True)
        comptime for device in range(num_devices):
            comptime for i in range(num_devices):
                out_tensors[device * num_devices + i] = out_tiles[i]

        # The peer arrays are indexed by global rank. Without an explicit
        # rank, the device id stands in for it, which is right only when the
        # world is devices 0..N-1 in order; under recording the context
        # reports the builder's device.
        var my_rank = peer_rank.or_else(Int(ctx.device().id))
        allgather[ngpus=num_devices](
            in_tensors, out_tensors, rank_sigs, ctx, my_rank
        )

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        peer_rank: Optional[Int] = None,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        ctx: DeviceContext,
    ) raises:
        """Per-device allgather through live addresses.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name used for profiling and debugging.
            peer_rank: THIS GPU's index in `inputs` and `signal_buffers`;
                `None` takes it from the device id.

        Args:
            outputs: THIS GPU's gathered pieces, one per peer in peer order.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()

        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[dtype]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[dtype]().as_imm()
            )

        comptime assert conforms_to(type_of(outputs[0]), TileTensorable)
        comptime OutputTile = type_of(
            TileTensor(
                ptr=outputs[0]
                .to_tile_tensor[dtype]()
                .unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                layout=row_major(Int(0)),
            )
        )

        def output_tile[i: Int]() {imm outputs} -> OutputTile:
            comptime assert conforms_to(type_of(outputs[i]), TileTensorable)
            return rebind[OutputTile](
                TileTensor(
                    ptr=outputs[i]
                    .to_tile_tensor[dtype]()
                    .unsafe_ptr()
                    .unsafe_origin_cast[MutUntrackedOrigin](),
                    layout=row_major(Int(outputs[i].shape().product())),
                )
            )

        Self._launch[peer_rank=peer_rank](
            in_tensors,
            Array[OutputTile, num_devices](fill_with_unrolled=output_tile),
            rank_sigs,
            Int(signal_buffers[0].shape().product()),
            ctx,
        )

    @staticmethod
    def build_graph[
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor & Stable, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        peer_rank: Optional[Int] = None,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device allgather into a device graph.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name used for profiling and debugging.
            peer_rank: THIS GPU's index in `inputs` and `signal_buffers`;
                `None` takes it from the device id.

        Args:
            outputs: THIS GPU's gathered pieces, one per peer in peer order.
            inputs: Slot-backed input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            builder: Device graph builder for THIS GPU.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()

        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = type_of(inputs[0].to_tile_tensor[dtype]())
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[dtype]()
            )

        comptime assert conforms_to(type_of(outputs[0]), TileTensorable)
        comptime OutputTile = type_of(
            TileTensor(
                ptr=outputs[0]
                .to_tile_tensor[dtype]()
                .unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                layout=row_major(Int(0)),
            )
        )

        def output_tile[i: Int]() {imm outputs} -> OutputTile:
            comptime assert conforms_to(type_of(outputs[i]), TileTensorable)
            return rebind[OutputTile](
                TileTensor(
                    ptr=outputs[i]
                    .to_tile_tensor[dtype]()
                    .unsafe_ptr()
                    .unsafe_origin_cast[MutUntrackedOrigin](),
                    layout=row_major(Int(outputs[i].shape().product())),
                )
            )

        Self._launch[peer_rank=peer_rank](
            in_tensors,
            Array[OutputTile, num_devices](fill_with_unrolled=output_tile),
            rank_sigs,
            Int(signal_buffers[0].shape().product()),
            builder.recording_context(),
        )


@extensibility.register("mo.composite.bundled.allreduce_add_rms_norm_quant_fp8")
struct BundledAllReduceAddRMSNormQuantFP8:
    """Registers the `mo.composite.bundled.allreduce_add_rms_norm_quant_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
    ](
        output: Some[Output & Tensor],
        out_scale: Some[Output & Tensor],
        out_residual: Some[Output & Tensor],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        residual: Some[Input & Tensor],
        gamma: Some[Input & Tensor],
        scale_ub: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Per-device fused allreduce.sum + add + rms_norm + fp8 quantize.

        Single-device analog of `DistributedAllReduceAddRMSNormQuantFP8`, for
        use inside `mo.parallel`. The parallel framework launches one
        instance per GPU; this kernel invokes the same underlying primitive
        (`allreduce_residual_rmsnorm`) that the distributed variant calls
        from within `_launch_device_collective`, but for a single device.

        Parameters:
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name used for profiling and debugging.
            epsilon: RMSNorm epsilon scalar.
            weight_offset: RMSNorm weight offset scalar.

        Args:
            output: FP8 quantized output tensor for THIS GPU.
            out_scale: Per-token scale tensor for THIS GPU.
            out_residual: Post-add residual tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            residual: Residual tensor for THIS GPU.
            gamma: RMSNorm weight for THIS GPU.
            scale_ub: Quantization scale upper bound scalar (host).
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime output_type = type_of(output).dtype
        comptime scales_type = type_of(out_scale).dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert type_of(output).rank == rank
        comptime assert type_of(out_scale).rank == rank
        comptime assert type_of(out_residual).dtype == dtype
        comptime assert type_of(out_residual).rank == rank
        comptime assert type_of(residual).dtype == dtype
        comptime assert type_of(residual).rank == rank
        comptime assert type_of(gamma).dtype == dtype
        comptime assert type_of(gamma).rank == 1
        comptime assert type_of(scale_ub).dtype == DType.float32

        # Logic copied from kernel host code
        var in_num_elems = Int(inputs[0].shape().product())
        comptime last_dim_idx = rank - 1
        var cols = Int(inputs[0].shape()[last_dim_idx].value())
        var rows = in_num_elems // cols
        var rows_per_rank = ceildiv(rows, num_devices)

        # Output scratch holds fp8 (1 byte) when quantizing; this op is
        # FP8-only, but size by output_type so the math stays correct if the
        # output ever matches the input dtype (no-quant path).
        var output_size_bytes = cols * rows_per_rank * size_of[output_type]()
        var pessimistic_simd_width = 32  # just to be safe...
        var scales_size_bytes = (
            align_up(
                rows_per_rank * size_of[scales_type](), pessimistic_simd_width
            ) if output_type
            != dtype else 0
        )
        var residual_size_bytes = cols * rows_per_rank * size_of[dtype]()

        var scratch_buffer_size_bytes = (
            output_size_bytes + scales_size_bytes + residual_size_bytes
        )
        _check_signal_buffer_size(
            Int(signal_buffers[0].shape().product()),
            scratch_buffer_size_bytes,
        )

        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[dtype]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)

        # A collective addresses the signal buffers through flat pointers; a
        # non-pointer storage policy aborts here at run time.
        def rank_sig[
            i: Int
        ]() {imm signal_buffers} -> Pointer[Signal, MutAnyOrigin]:
            comptime assert conforms_to(
                type_of(signal_buffers[i]), TileTensorable
            )
            return (
                signal_buffers[i]
                .to_tile_tensor()
                .ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[dtype]().as_imm()
            )

        comptime assert conforms_to(type_of(residual), TileTensorable)
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(out_residual), TileTensorable)
        comptime assert conforms_to(type_of(gamma), TileTensorable)
        comptime assert conforms_to(type_of(scale_ub), TileTensorable)
        comptime assert conforms_to(type_of(out_scale), TileTensorable)
        allreduce_residual_rmsnorm(
            in_tensors,
            residual.to_tile_tensor[dtype]().as_imm(),
            output.to_tile_tensor[output_type](),
            out_residual.to_tile_tensor[dtype](),
            gamma.to_tile_tensor[dtype](),
            epsilon,
            weight_offset.cast[dtype](),
            scale_ub.to_tile_tensor[DType.float32]().ptr[],
            out_scale.to_tile_tensor[scales_type](),
            rank_sigs,
            ctx,
        )
