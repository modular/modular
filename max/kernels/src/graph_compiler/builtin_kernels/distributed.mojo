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

from std.math import ceildiv
from std.sys import get_defined_bool
from std.sys.info import size_of
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from comm.allgather import allgather
from comm.allreduce import allreduce

from comm.allreduce_residual_rmsnorm import allreduce_residual_rmsnorm
from comm.reducescatter import reducescatter
from comm.broadcast import broadcast
from comm.scatter import scatter
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from std.gpu.host import DeviceContextList
from layout.tile_tensor import row_major
from layout import Coord, TileTensor, coord_to_index_list, row_major
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    OutputVariadicTensors,
)
from extensibility import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from std.memory import UnsafePointer
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _check_signal_buffer_size,
    _launch_device_collective,
    _partitioned_scratch_requirement,
)


@compiler.register("mo.distributed.allreduce.sum")
struct DistributedAllReduceSum:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Distributed allreduce operation implementation for sum reduction.

        Args:
            outputs: Output tensors (one per GPU) to store reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width (per allreduce.mojo)
            - Requires identical tensor shapes across all participating GPUs
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # allreduce 2-stage uses size/ngpus scratch space
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](inputs[0].size())
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # output_lambda writes each device's reduced output into the fused
        # epilogue output tensor.  Defined at execute scope so that
        # epilogue_wrapper in vendor_ccl.allreduce (also execute scope) can
        # call it without triggering the MLIR 'kgen.param.declare.region must
        # have subprogram scope' error that arises when parameterized functions
        # are defined inside closures.
        @always_inline
        @parameter
        def output_lambda[
            output_index: Int,
            _dtype: DType,
            _width: SIMDSize,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            outputs[output_index]._lambda_store[
                width=_width, element_alignment=_alignment
            ](
                rebind[IndexList[rank]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        # Marshal signal buffers into the expected format.
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)
        comptime for i in range(num_devices):
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
            logger.info("Executing: Vendor CCL")
            comptime InputTensorType = type_of(
                inputs[0].to_tile_tensor[DType.int64]().as_immut()
            )
            var in_tensors = InlineArray[InputTensorType, num_devices](
                uninitialized=True
            )
            comptime for i in range(num_devices):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[i].to_tile_tensor[DType.int64]().as_immut()
                )

            @always_inline
            def launch_vendor_allreduce[
                index: Int
            ]() raises {
                read in_tensors,
                read rank_sigs,
                read dev_ctxs_input,
                read outputs,
            }:
                # _get_global_comms has a check-then-create race: two
                # threads seeing null simultaneously would both call
                # ncclCommInitAll and leak one set of communicators.
                # Only device 0 initializes; others spin-wait.
                comptime if index == 0:
                    vendor_ccl.init_comms(num_devices)
                else:
                    vendor_ccl.wait_for_comms(num_devices)

                vendor_ccl.allreduce[
                    ngpus=num_devices,
                    output_lambda=output_lambda[output_index=index, ...],
                ](
                    in_tensors,
                    outputs[index].to_tile_tensor[DType.int64](),
                    rank_sigs,
                    dev_ctxs_input[index],
                )

            _launch_device_collective[num_devices](
                launch_vendor_allreduce, dev_ctxs_input
            )
            return

        # Custom allreduce path.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, inputs.size](
            uninitialized=True
        )
        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )

        @always_inline
        def launch_allreduce[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            allreduce[
                ngpus=num_devices,
                output_lambda=output_lambda[output_index=index, ...],
            ](
                in_tensors,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[num_devices](launch_allreduce, dev_ctxs_input)


@compiler.register("mo.distributed.reducescatter.sum")
struct DistributedReduceScatterSum:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        axis: Int = 0,
        group_size: Int = 0,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Distributed reduce-scatter operation implementation for sum reduction.

        Args:
            outputs: Output tensors (one per GPU) to store scattered reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width
            - Requires identical tensor shapes within each reduce-scatter group
        """
        comptime num_devices = inputs.size
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"
        comptime assert group_size >= 1, "group_size must be at least 1"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Reduce-scatter doesn't use scratch storage, so
        # only need enough signal_buffer space for Signal struct
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Marshal input tensors into fully dynamic TileTensors so groups can
        # have different static shapes while sharing one InlineArray type.

        @always_inline
        def launch_reducescatter[
            index: Int
        ]() raises {
            read inputs,
            read signal_buffers,
            read dev_ctxs_input,
            read outputs,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            # Full-world collectives keep scope 0; grouped collectives get a
            # distinct nonzero scope per device-group so their barrier counters
            # never poison the full-world bank on the shared Signal buffers.
            comptime domain_id = 0 if group_size == num_devices else group_size
            comptime InputTensorType = type_of(
                inputs[group_start].to_tile_tensor[DType.int64]().as_immut()
            )

            var in_tensors = InlineArray[InputTensorType, group_size](
                uninitialized=True
            )
            var rank_sigs = InlineArray[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for i in range(group_size):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[group_start + i]
                    .to_tile_tensor[DType.int64]()
                    .as_immut()
                )

                rank_sigs[i] = (
                    signal_buffers[group_start + i]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            @always_inline
            @parameter
            def output_lambda[
                output_index: Int,
                _dtype: DType,
                _width: SIMDSize,
                *,
                _alignment: Int,
            ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
                outputs[output_index]._lambda_store[
                    width=_width,
                    element_alignment=_alignment,
                ](
                    rebind[IndexList[rank]](coord_to_index_list(coords)),
                    rebind[SIMD[dtype, _width]](val),
                )

            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            reducescatter[
                ngpus=group_size,
                output_lambda=output_lambda[output_index=index, ...],
                axis=axis,
                domain_id=domain_id,
            ](
                in_tensors,
                out_buf.make_dynamic[DType.int64](),
                rank_sigs,
                dev_ctxs_input[index],
                local_rank=local_rank,
            )

        _launch_device_collective[num_devices](
            launch_reducescatter, dev_ctxs_input
        )


@compiler.register("mo.distributed.allgather")
struct DistributedAllGather:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        group_size: Int = 0,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Distributed allgather operation implementation.

        Args:
            outputs: Output tensors (one per GPU) to store gathered results.
            inputs: Input tensors (one per GPU) containing values to gather.
            signal_buffers: Device buffer values used for synchronization.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        comptime num_devices = inputs.size
        comptime assert (
            signal_buffers.size == num_devices
            and outputs.size == num_devices * group_size
        ), (
            "expected allgather inputs, signal buffers to have the same"
            " number of elements and outputs to have num_devices * group_size"
        )
        comptime assert group_size >= 1, "group_size must be at least 1"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        var scratch_buffer_size_bytes = 0  # no allgather impl uses scratch
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        @always_inline
        def launch_allgather[
            index: Int
        ]() raises {
            read inputs,
            read outputs,
            read signal_buffers,
            read dev_ctxs_input,
        }:
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size
            # Full-world collectives keep domain 0; grouped collectives get a
            # distinct nonzero domain per device-group so their barrier counters
            # never poison the full-world bank on the shared Signal buffers.
            comptime domain_id = 0 if group_size == num_devices else group_size
            comptime InputTensorType = type_of(
                TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                        inputs[group_start]._ptr
                    ),
                    row_major(inputs[group_start].size()),
                )
            )
            comptime OutputTensorType = type_of(
                TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                        outputs[index * group_size]._ptr
                    ),
                    row_major(outputs[index * group_size].size()),
                )
            )
            var group_in_tensors = InlineArray[InputTensorType, group_size](
                uninitialized=True
            )
            var device_out_tensors = InlineArray[OutputTensorType, group_size](
                uninitialized=True
            )
            var group_rank_sigs = InlineArray[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)

            comptime for src_idx in range(group_size):
                group_in_tensors[src_idx] = TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                        inputs[group_start + src_idx]._ptr
                    ),
                    row_major(inputs[group_start + src_idx].size()),
                )
                device_out_tensors[src_idx] = TileTensor(
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                        outputs[index * group_size + src_idx]._ptr
                    ),
                    row_major(outputs[index * group_size + src_idx].size()),
                )
                group_rank_sigs[src_idx] = (
                    signal_buffers[group_start + src_idx]
                    ._ptr.bitcast[Signal]()
                    .as_unsafe_any_origin()
                )

            allgather[ngpus=group_size, domain_id=domain_id](
                group_in_tensors,
                device_out_tensors,
                group_rank_sigs,
                dev_ctxs_input[index],
                local_rank,
            )

        _launch_device_collective[num_devices](launch_allgather, dev_ctxs_input)


@compiler.register("mo.distributed.broadcast")
struct DistributedBroadcast:
    """Distributed broadcast: copy tensor from root GPU to all GPUs.

    A single instance of this op handles all participating GPUs. It receives:
    - input: The source tensor from the root GPU (P2P accessible)
    - outputs: Destination tensors, one per GPU
    - signal_buffers: Synchronization buffers for all participating GPUs
    - dev_ctxs_input: Device contexts for all participating GPUs
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Execute distributed broadcast operation.

        Parameters:
            dtype: Data type of the tensor.
            rank: Tensor rank (number of dimensions).
            root: Index of the root GPU (source of data).
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.

        Args:
            outputs: Output tensors (one per GPU) to store broadcast results.
            input: Input tensor from root GPU (P2P accessible from all GPUs).
            signal_buffers: Synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (MAX_GPUS).
            - Requires P2P access between GPUs (NVLink or PCIe P2P).
        """
        comptime num_devices = outputs.size
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"
        comptime assert (
            root >= 0 and root < num_devices
        ), "root GPU index must be in range [0, ngpus)"

        # 2-stage broadcast stages 1/ngpus of input into each signal buffer payload.
        # 1-stage broadcast doesn't use payload at all (direct P2P from root).
        # Use 2-stage requirement as upper bound.
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input.size())
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        var in_buf = input.to_tile_tensor[DType.int64]()

        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(signal_buffers.size):
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        def launch_broadcast[
            index: Int
        ]() raises {
            read in_buf,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            var out_buf = TileTensor[mut=True](
                outputs[index]
                .to_tile_tensor[DType.int64]()
                .make_dynamic[DType.int64]()
                .ptr,
                in_buf.layout,
            )
            broadcast[num_devices](
                in_buf,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
                root,
            )

        _launch_device_collective[num_devices](launch_broadcast, dev_ctxs_input)


@compiler.register("mo.distributed.scatter")
struct DistributedScatter:
    """Distributed scatter: send different chunks to different device groups.

    Each DP replica group receives a different input chunk from the root GPU.
    All TP devices within the same replica get the same chunk via P2P pull.

    This op receives ngpus input tensors (one per GPU, padded from dp_size
    distinct chunks) plus ngpus signal buffers for synchronization. All GPUs
    see all chunks so they compute the same grid size (avoiding barrier
    deadlocks).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        comptime ngpus = signal_buffers.size
        comptime assert (
            root >= 0 and root < ngpus
        ), "root GPU index must be in range [0, ngpus)"
        comptime assert inputs.size == ngpus, (
            "expected scatter inputs and signal buffers to have"
            " the same number of elements"
        )

        # Scatter uses signal buffers for barriers only (no payload staging),
        # so payload_size=0. This still validates the buffer holds a Signal.
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Inputs can have different static shapes, so use make_dynamic to
        # produce a homogeneous fully-dynamic TileTensor type for InlineArray.
        comptime InputTensorType = type_of(
            inputs[0]
            .to_tile_tensor[DType.int64]()
            .make_dynamic[DType.int64]()
            .as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, ngpus](uninitialized=True)
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(ngpus):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i]
                .to_tile_tensor[DType.int64]()
                .make_dynamic[DType.int64]()
                .as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        def launch_scatter[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            scatter[ngpus=ngpus, dp_size=ngpus](
                in_tensors,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[ngpus](launch_scatter, dev_ctxs_input)


@compiler.register("mo.composite.distributed.allreduce_add_rms_norm_quant_fp8")
struct DistributedAllReduceAddRMSNormQuantFP8:
    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=output_type, rank=rank, ...],
        outputs_scales: OutputVariadicTensors[
            dtype=scales_type, rank=rank, ...
        ],
        outputs_residual: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        residuals: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        gammas: InputVariadicTensors[dtype=dtype, rank=1, ...],
        epsilons: InputVariadicTensors[dtype=dtype, ...],
        weight_offsets: InputVariadicTensors[dtype=dtype, ...],
        scales_ub: InputVariadicTensors[dtype=DType.float32, ...],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # Logic copied from kernel host code
        # Note: this is a prime candidate for a method on a kernel
        # struct which advertises kernel info to the GC!
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

        # Filter the dev_ctxs_list to have only the GPU devices.
        # The kernel also takes CPU operands, so CPU devices must be removed.
        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        # Marshal input tensors into TileTensors.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, inputs.size](
            uninitialized=True
        )

        # Marshal signal buffers.
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(inputs.size):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        def launch_fused_allreduce[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs,
            read gammas,
            read epsilons,
            read weight_offsets,
            read scales_ub,
            read outputs,
            read outputs_scales,
            read outputs_residual,
            read residuals,
        }:
            # Marshal per-device outputs and residual as TileTensors.
            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            var out_scales_buf = outputs_scales[index].to_tile_tensor[
                DType.int64
            ]()
            var out_residual_buf = outputs_residual[index].to_tile_tensor[
                DType.int64
            ]()
            var residual_buf = (
                residuals[index].to_tile_tensor[DType.int64]().as_immut()
            )
            var gamma_tensor = gammas[index].to_tile_tensor[DType.int64]()

            # TODO: Add a new struct like `VariadicInputScalar`` to
            # represent instead of manually loading the values in the
            # kernel code.
            var epsilon = epsilons[index].unsafe_ptr()[]
            var weight_offset = weight_offsets[index].unsafe_ptr()[]
            var scale_ub = scales_ub[index].unsafe_ptr()[]

            allreduce_residual_rmsnorm(
                in_tensors,
                residual_buf,
                out_buf,
                out_residual_buf,
                gamma_tensor,
                epsilon,
                weight_offset,
                scale_ub,
                out_scales_buf,
                rank_sigs,
                dev_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_fused_allreduce, dev_ctxs)
