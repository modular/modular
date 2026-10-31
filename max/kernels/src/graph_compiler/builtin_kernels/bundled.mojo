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
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    OutputVariadicTensors,
    StableInputTensor,
    StableInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceContext, DeviceGraphBuilder

from .kernels import _check_signal_buffer_size, _partitioned_scratch_requirement


@extensibility.register("mo.bundled.allreduce.sum")
struct BundledAllReduceSum:
    """Registers the `mo.bundled.allreduce.sum` graph op with the graph compiler.
    """

    @staticmethod
    def _launch[
        dtype: DType,
        rank: Int,
        num_devices: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        //,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        in_tensors: Array[
            TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
            num_devices,
        ],
        input_size: Int,
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # allreduce 2-stage uses size/ngpus scratch space
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input_size)
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        var out_layout = output.runtime_layout()
        var rank_sigs = Array[_, num_devices](
            fill_with_unrolled=lambda [i: Int]() -> Pointer[
                Signal, MutAnyOrigin
            ]: (
                signal_buffers[i]
                ._ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        )

        @inline(.always)
        def output_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) {var output} -> None:
            output._lambda_store[width=_width, element_alignment=alignment](
                coords,
                rebind[SIMD[dtype, _width]](val),
            )

        allreduce[ngpus=num_devices](
            in_tensors,
            out_layout,
            output_lambda,
            rank_sigs,
            ctx,
        )

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        """Per-device allreduce sum, for use with mo.parallel dispatch.

        This path does not take the kernel-load gate: `mo.parallel` emits N
        independent async stubs, with no single point to claim or release it,
        so a first-time kernel load can still interleave with these launches.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = inputs.size
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_imm()
            )

        Self._launch(output, in_tensors, inputs[0].size(), signal_buffers, ctx)

    @staticmethod
    def build_graph[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: StableInputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device allreduce sum into a device graph.

        The peers arrive as stable address slots: the recorded kernels read
        each peer's data pointer out of its slot at replay time, so the graph
        does not bake in the peer buffers themselves.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Slot-backed input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            builder: Device graph builder for THIS GPU.
        """
        var in_tensors = inputs.to_tile_tensor()

        # The recording context reports the builder's backing device id, so
        # the rank resolves exactly as it does under `execute`.
        Self._launch(
            output,
            in_tensors,
            inputs[0].size(),
            signal_buffers,
            builder.recording_context(),
        )


@extensibility.register("mo.bundled.broadcast")
struct BundledBroadcast:
    """Registers the `mo.bundled.broadcast` graph op with the graph compiler.

    The per-device form of `mo.distributed.broadcast`: one instance per GPU
    copies the root device's tensor into its own output. `execute` reads the
    root tensor through its live data pointer; `build_graph` reads it through
    a stable address slot (`StableInputTensor`), so the recorded device graph
    replays against whatever buffer the slot names at the time.
    """

    @staticmethod
    def _launch[
        dtype: DType,
        rank: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        //,
        root: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        in_buf: TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
        input_size: Int,
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        comptime num_devices = signal_buffers.size
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input_size)
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        var rank_sigs = Array[_, num_devices](
            fill_with_unrolled=lambda [i: Int]() -> Pointer[
                Signal, MutAnyOrigin
            ]: (
                signal_buffers[i]
                ._ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        )
        var out_buf = TileTensor[mut=True](
            output.to_tile_tensor[.int64]().make_dynamic[.int64]()._storage,
            in_buf.layout,
        )

        # The signal buffers are indexed by device id, which is the rank the
        # context reports; under recording it reports the builder's device.
        broadcast[num_devices](
            in_buf, out_buf, rank_sigs, ctx, root, rank=Int(ctx.device().id)
        )

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        """Per-device broadcast through live addresses.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            root: Index of the root device in the signal buffer list.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            input: The root GPU's tensor.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        Self._launch[root=root](
            output,
            input.to_tile_tensor[.int64](),
            input.size(),
            signal_buffers,
            ctx,
        )

    @staticmethod
    def build_graph[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: StableInputTensor[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device broadcast into a device graph.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            root: Index of the root device in the signal buffer list.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            input: The root GPU's tensor, read through its stable slot.
            signal_buffers: Signal buffers for ALL participating GPUs.
            builder: Device graph builder for THIS GPU.
        """
        Self._launch[root=root](
            output,
            input.to_tile_tensor(),
            input.size(),
            signal_buffers,
            builder.recording_context(),
        )


@extensibility.register("mo.bundled.allgather")
struct BundledAllGather:
    """Registers the `mo.bundled.allgather` graph op with the graph compiler.

    The per-device form of `mo.distributed.allgather`, full world only: one
    instance per GPU gathers every peer's tensor into its own outputs, one
    per peer in peer order. `execute` reads the peers through live data
    pointers; `build_graph` reads them through stable address slots
    (`StableInputVariadicTensors`).
    """

    @staticmethod
    def _launch[
        dtype: DType,
        rank: Int,
        num_devices: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        //,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        in_tensors: Array[
            TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
            num_devices,
        ],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            outputs.size == num_devices and signal_buffers.size == num_devices
        ), "expected one output and one signal buffer per peer"
        _check_signal_buffer_size(signal_buffers[0].size(), 0)

        var rank_sigs = Array[_, num_devices](
            fill_with_unrolled=lambda [i: Int]() -> Pointer[
                Signal, MutAnyOrigin
            ]: (
                signal_buffers[i]
                ._ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        )

        # The kernel takes every device's outputs, laid out
        # `[device * num_devices + source]`, but a full-world gather writes
        # only this device's slice, so the peers' slots alias it.
        comptime OutputTensorType = type_of(
            TileTensor(
                rebind[Pointer[Scalar[dtype], MutAnyOrigin]](outputs[0]._ptr),
                row_major(outputs[0].size()),
            )
        )
        var out_tensors = Array[OutputTensorType, num_devices * num_devices](
            uninitialized=True
        )
        comptime for device in range(num_devices):
            comptime for i in range(num_devices):
                out_tensors[device * num_devices + i] = TileTensor(
                    rebind[Pointer[Scalar[dtype], MutAnyOrigin]](
                        outputs[i]._ptr
                    ),
                    row_major(outputs[i].size()),
                )

        # The signal buffers are indexed by device id, which is the rank the
        # context reports; under recording it reports the builder's device.
        allgather[ngpus=num_devices](
            in_tensors, out_tensors, rank_sigs, ctx, Int(ctx.device().id)
        )

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        """Per-device allgather through live addresses.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            outputs: THIS GPU's gathered pieces, one per peer in peer order.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = inputs.size
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_imm()
            )

        Self._launch(outputs, in_tensors, signal_buffers, ctx)

    @staticmethod
    def build_graph[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: StableInputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device allgather into a device graph.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            outputs: THIS GPU's gathered pieces, one per peer in peer order.
            inputs: Slot-backed input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            builder: Device graph builder for THIS GPU.
        """
        Self._launch(
            outputs,
            inputs.to_tile_tensor(),
            signal_buffers,
            builder.recording_context(),
        )


@extensibility.register("mo.composite.bundled.allreduce_add_rms_norm_quant_fp8")
struct BundledAllReduceAddRMSNormQuantFP8:
    """Registers the `mo.composite.bundled.allreduce_add_rms_norm_quant_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
    ](
        output: OutputTensor[dtype=output_type, rank=rank, ...],
        out_scale: OutputTensor[dtype=scales_type, rank=rank, ...],
        out_residual: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        residual: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        scale_ub: InputTensor[dtype=DType.float32, ...],
        ctx: DeviceContext,
    ) raises:
        """Per-device fused allreduce.sum + add + rms_norm + fp8 quantize.

        Single-device analog of `DistributedAllReduceAddRMSNormQuantFP8`, for
        use inside `mo.parallel`. The parallel framework launches one
        instance per GPU; this kernel invokes the same underlying primitive
        (`allreduce_residual_rmsnorm`) that the distributed variant calls
        from within `_launch_device_collective`, but for a single device.

        Parameters:
            dtype: Element type of the input and residual tensors.
            output_type: Element type of the quantized output tensor.
            scales_type: Element type of the per-token scale tensor.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
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
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # Logic copied from kernel host code
        var in_num_elems = inputs[0].size()
        comptime last_dim_idx = type_of(inputs[0]).rank - 1
        var cols = inputs[0].dim_size[last_dim_idx]()
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
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        var rank_sigs = Array[_, num_devices](
            fill_with_unrolled=lambda [i: Int]() -> Pointer[
                Signal, MutAnyOrigin
            ]: (
                signal_buffers[i]
                ._ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        )

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_imm()
            )

        allreduce_residual_rmsnorm(
            in_tensors,
            residual.to_tile_tensor[.int64]().as_imm(),
            output.to_tile_tensor[.int64](),
            out_residual.to_tile_tensor[.int64](),
            gamma.to_tile_tensor[.int64](),
            epsilon,
            weight_offset.cast[dtype](),
            scale_ub.unsafe_ptr()[],
            out_scale.to_tile_tensor[.int64](),
            rank_sigs,
            ctx,
        )
