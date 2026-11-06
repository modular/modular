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

"""Registers distributed and multi-GPU collective graph ops backed by the `comm` and `shmem` kernels."""

from std.math.uutils import ualign_down
from std.math import align_down
from std.sys import get_defined_bool
from std.sys.info import align_of, size_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from comm.allgather import allgather, allgather_rows_slice
from comm.allreduce import allreduce
from comm.alltoall import all_to_all

from comm.allreduce_lamport_rmsnorm import lamport_allreduce_rmsnorm
from comm.allreduce_residual_rmsnorm import (
    _dispatch_ar_norm,
    allreduce_residual_rmsnorm,
)
from comm.allgather_rmsnorm import (
    AG_QUANT_ONLY_THRESHOLD,
    _dispatch_ag_norm,
    _dispatch_ag_norm_quant,
    allgather_gated_residual_rmsnorm,
    allgather_rmsnorm,
)
from linalg.block_scaled_quantization import (
    quantize_mx_amd,
    quantize_mxfp8_lane_group,
)
from linalg.fp6_quantization import (
    quantize_mxfp6_amd,
    quantize_mxfp6_lane_group,
)
from linalg.fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    pack_fp6_x4,
)
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from comm.lamport import Lamport
from max.gpu import WARP_SIZE
from comm.reducescatter import ReduceScatterConfig, reducescatter
from comm.reducescatter_rmsnorm import _dispatch_rs_norm, reducescatter_rmsnorm
from comm.reducescatter_quant import (
    RS_QUANT_GROUP,
    reducescatter_quant,
    rs_quant_scratch_bytes,
)
from nn.normalization import rms_norm_gpu
from comm.broadcast import broadcast
from comm.device_collective import _launch_device_collective
from comm.scatter import scatter
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from max.gpu.host import DeviceContext, DeviceContextArray
from max.gpu.primitives.grid_controls import PDLLevel
from layout.tile_tensor import row_major
from layout import Coord, TileTensor, row_major
from layout.coord import DynamicCoord
from layout.layout import Layout
from extensibility import (
    Fused,
    Input,
    InputTensor,
    InputVariadicTensors,
    MutableInput,
    Output,
    OutputTensor,
    OutputVariadicTensors,
    Storable,
    StoreAccess,
    Tensor,
    TensorTuple,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils.index import Index
from std.collections import Array, Optional

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _check_signal_buffer_size,
    _check_tensor_tuple,
    _partitioned_scratch_requirement,
)


def _check_uniform_cols[
    dtype: DType, rank: Int, //, op: StaticString
](inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...]) raises:
    """Rejects a world whose devices disagree on the column count.

    The fused norm collectives size every group's windows and fuse gate from
    device 0's columns, and the dynamic world-view arrays accept any shape.
    """
    comptime last_dim_idx = rank - 1
    comptime static_cols = Int(inputs.static_specs[0].shape_tuple[last_dim_idx])
    comptime for i in range(1, inputs.size):
        comptime static_cols_i = Int(
            inputs.static_specs[i].shape_tuple[last_dim_idx]
        )
        comptime assert (
            static_cols < 0 or static_cols_i < 0 or static_cols == static_cols_i
        ), (
            "fused norm collectives require the same column count on every"
            " device; device groups may differ only in rows"
        )
    var cols = inputs[0].dim_size(last_dim_idx)
    comptime for i in range(1, inputs.size):
        var cols_i = inputs[i].dim_size(last_dim_idx)
        if cols_i != cols:
            raise Error(
                String(
                    op,
                    (
                        " requires the same column count on every device:"
                        " device 0 has "
                    ),
                    cols,
                    " columns, device ",
                    i,
                    " has ",
                    cols_i,
                )
            )


def _check_uniform_cols[
    op: StaticString, *Ts: Tensor
](inputs: TensorTuple[*Ts]) raises:
    """Rejects a world whose devices disagree on the column count.

    The fused norm collectives size every group's windows and fuse gate from
    device 0's columns, and the dynamic world-view arrays accept any shape.
    """
    comptime last_dim_idx = Ts[0].rank - 1
    comptime static_cols = Ts[0]._shape_types[last_dim_idx].static_value
    comptime for i in range(1, len(Ts)):
        comptime static_cols_i = Ts[i]._shape_types[last_dim_idx].static_value
        comptime assert (
            static_cols < 0 or static_cols_i < 0 or static_cols == static_cols_i
        ), (
            "fused norm collectives require the same column count on every"
            " device; device groups may differ only in rows"
        )
    var cols = Int(inputs[0].shape()[last_dim_idx].value())
    comptime for i in range(1, len(Ts)):
        var cols_i = Int(inputs[i].shape()[last_dim_idx].value())
        if cols_i != cols:
            raise Error(
                String(
                    op,
                    (
                        " requires the same column count on every device:"
                        " device 0 has "
                    ),
                    cols,
                    " columns, device ",
                    i,
                    " has ",
                    cols_i,
                )
            )


@extensibility.register("mo.distributed.allreduce.sum")
struct DistributedAllReduceSum:
    """Registers the `mo.distributed.allreduce.sum` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Fused & StoreAccess, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        group_size: Int = 0,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Distributed allreduce operation implementation for sum reduction.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name for profiling.
            group_size: Number of devices per independent allreduce group;
                must evenly divide the total number of devices. Zero (the
                attribute default) means all devices form one group.

        Args:
            outputs: Output tensors (one per GPU) to store reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width (per allreduce.mojo)
            - Requires identical tensor shapes within each allreduce group
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime effective_group_size = (
            num_devices if group_size == 0 else group_size
        )
        comptime assert (
            effective_group_size >= 1
        ), "group_size must be at least 1"
        comptime assert (
            num_devices % effective_group_size == 0
        ), "group_size must evenly divide the number of devices"

        # allreduce 2-stage uses size/ngpus scratch space; check per group
        # since groups may carry different row counts.
        comptime for g in range(num_devices // effective_group_size):
            comptime group_start = g * effective_group_size
            var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
                effective_group_size, dtype
            ](Int(inputs[group_start].shape().product()))
            _check_signal_buffer_size(
                Int(signal_buffers[group_start].shape().product()),
                scratch_buffer_size_bytes,
            )

        # Marshal signal buffers into the expected format. World-view, indexed
        # by GLOBAL device rank -- the collectives below do their own
        # group-local slicing internally.
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

        # World-view input tensors, indexed by GLOBAL device rank. Built once
        # (shared by every per-device launch below) since it does not depend
        # on which group a device belongs to.
        comptime InputTile = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]

        def input_tile[i: Int]() {imm inputs} -> InputTile:
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            return rebind[InputTile](
                inputs[i].to_tile_tensor().make_dynamic[.int64]()
            )

        var in_tensors = Array[InputTile, num_devices](
            fill_with_unrolled=input_tile
        )
        var output_views = TensorTuple._fused_view(outputs)

        comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
            comptime assert (
                effective_group_size == num_devices
            ), "grouped allreduce is not supported on the vendor CCL path"
            logger.info("Executing: Vendor CCL")

            @inline(.always)
            def launch_vendor_allreduce[
                index: Int
            ]() raises {
                imm in_tensors,
                imm rank_sigs,
                imm dev_ctxs_input,
                imm outputs,
                imm output_views,
            }:
                # _get_global_comms has a check-then-create race: two
                # threads seeing null simultaneously would both call
                # ncclCommInitAll and leak one set of communicators.
                # Only device 0 initializes; others spin-wait.
                comptime if index == 0:
                    vendor_ccl.init_comms(num_devices)
                else:
                    vendor_ccl.wait_for_comms(num_devices)

                # Writes this device's reduced output into its fused epilogue
                # output. It runs on the device, so it copies the views.
                @inline(.always)
                def output_lambda[
                    _dtype: DType,
                    _width: SIMDLength,
                    *,
                    alignment: Int,
                ](coords: Coord, val: SIMD[_dtype, _width]) {
                    var output_views
                } -> None:
                    comptime assert conforms_to(
                        type_of(output_views[index]), Storable
                    )
                    output_views[index].store[
                        width=_width, element_alignment=alignment
                    ](
                        coords,
                        rebind[
                            SIMD[type_of(output_views[index]).dtype, _width]
                        ](val),
                    )

                vendor_ccl.allreduce[ngpus=num_devices](
                    in_tensors,
                    row_major(outputs[index].shape()),
                    output_lambda,
                    rank_sigs,
                    dev_ctxs_input[index],
                )

            _launch_device_collective[num_devices](
                launch_vendor_allreduce, dev_ctxs_input.copy()
            )
            return

        # Custom allreduce path: hand the collective the whole world plus the
        # group width, and let it do its own group-local slicing internally.
        @inline(.always)
        def launch_allreduce[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm outputs,
            imm output_views,
        }:
            var out_layout = row_major(outputs[index].shape())

            # Writes this device's reduced output into its fused epilogue
            # output. It runs on the device, so it copies the views.
            @inline(.always)
            def output_lambda[
                _dtype: DType,
                _width: SIMDLength,
                *,
                alignment: Int,
            ](coords: Coord, val: SIMD[_dtype, _width]) {
                var output_views
            } -> None:
                comptime assert conforms_to(
                    type_of(output_views[index]), Storable
                )
                output_views[index].store[
                    width=_width, element_alignment=alignment
                ](
                    coords,
                    rebind[SIMD[type_of(output_views[index]).dtype, _width]](
                        val
                    ),
                )

            allreduce[ngpus=num_devices, group_size=effective_group_size](
                in_tensors,
                out_layout,
                output_lambda,
                rank_sigs,
                dev_ctxs_input[index],
                my_rank=index,
            )

        _launch_device_collective[num_devices](
            launch_allreduce, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.reducescatter.sum")
struct DistributedReduceScatterSum:
    """Registers the `mo.distributed.reducescatter.sum` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Fused & StoreAccess, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        axis: Int = 0,
        group_size: Int = 0,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Distributed reduce-scatter operation implementation for sum reduction.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name for profiling.
            axis: Axis along which to scatter the reduced result
                (defaults to 0).
            group_size: Number of devices per reduce-scatter group; must be
                at least 1 and must evenly divide the total number of
                devices (defaults to 0).

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
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert group_size >= 1, "group_size must be at least 1"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Reduce-scatter doesn't use scratch storage, so
        # only need enough signal_buffer space for Signal struct
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            Int(signal_buffers[0].shape().product()), scratch_buffer_size_bytes
        )

        # World-view input tensors and signals, indexed by GLOBAL device rank;
        # `reducescatter` does its own group-local slicing internally.
        comptime InputTile = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]

        def input_tile[i: Int]() {imm inputs} -> InputTile:
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            return rebind[InputTile](
                inputs[i].to_tile_tensor().make_dynamic[.int64]()
            )

        var in_tensors = Array[InputTile, num_devices](
            fill_with_unrolled=input_tile
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
                .ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)

        # World-view output layouts, indexed by GLOBAL device rank; built once.
        # Every write goes through `output_lambda` below, so `reducescatter`
        # needs only each output's geometry.
        comptime OutputLayoutType = type_of(
            row_major(outputs[0].shape()).make_dynamic[.int64]()
        )
        var out_layouts = Array[OutputLayoutType, num_devices](
            uninitialized=True
        )
        comptime for i in range(num_devices):
            out_layouts[i] = rebind[OutputLayoutType](
                row_major(outputs[i].shape()).make_dynamic[.int64]()
            )
        var output_views = TensorTuple._fused_view(outputs)

        @inline(.always)
        def launch_reducescatter[
            index: Int
        ]() raises {
            imm in_tensors,
            imm out_layouts,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm output_views,
        }:
            # Writes this device's reduced output into its fused epilogue
            # output. It runs on the device, so it copies the views.
            @inline(.always)
            def output_lambda[
                _dtype: DType,
                _width: SIMDLength,
                *,
                alignment: Int,
            ](coords: Coord, val: SIMD[_dtype, _width]) {
                var output_views
            } -> None:
                comptime assert conforms_to(
                    type_of(output_views[index]), Storable
                )
                output_views[index].store[
                    width=_width, element_alignment=alignment
                ](
                    coords,
                    rebind[SIMD[type_of(output_views[index]).dtype, _width]](
                        val
                    ),
                )

            reducescatter[ngpus=num_devices, group_size=group_size, axis=axis](
                in_tensors,
                out_layouts,
                output_lambda,
                rank_sigs,
                dev_ctxs_input[index],
                my_rank=index,
            )

        _launch_device_collective[num_devices](
            launch_reducescatter, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.all_to_all")
struct DistributedAllToAll:
    """Registers the `mo.distributed.all_to_all` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        split_axis: Int,
        concat_axis: Int,
        group_size: Int,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Distributed all-to-all implementation.

        Each device splits its input into `group_size` chunks along
        `split_axis`, sends chunk `j` to its group's rank `j`, and
        concatenates the chunks it receives in source rank order along
        `concat_axis`.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name for profiling.
            split_axis: The input axis split into one chunk per peer.
            concat_axis: The output axis the received chunks concatenate on.
            group_size: Number of contiguous devices per all-to-all group;
                must be at least 2 and evenly divide the number of devices.

        Args:
            outputs: Output tensors, one per GPU.
            inputs: Input tensors, one per GPU; same shape within a group.
            signal_buffers: Preallocated synchronization buffers.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"
        comptime assert (
            0 <= split_axis < rank and 0 <= concat_axis < rank
        ), "all_to_all axes must be in range"

        var scratch_buffer_size_bytes = 0  # all_to_all uses no scratch
        _check_signal_buffer_size(
            Int(signal_buffers[0].shape().product()), scratch_buffer_size_bytes
        )

        comptime InPtr = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        var in_ptrs = Array[InPtr, num_devices](uninitialized=True)
        var out_ptrs = Array[
            MutPointer[Scalar[dtype], MutAnyOrigin], num_devices
        ](uninitialized=True)
        var num_elements = Array[Int, num_devices](uninitialized=True)
        var in_inner = Array[Int, num_devices](uninitialized=True)
        var out_inner = Array[Int, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            comptime assert conforms_to(type_of(outputs[i]), TileTensorable)
            in_ptrs[i] = rebind[InPtr](inputs[i].to_tile_tensor[dtype]().ptr)
            out_ptrs[i] = rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                outputs[i].to_tile_tensor[dtype]().ptr
            )
            var in_shape = inputs[i].shape()
            var out_shape = outputs[i].shape()
            num_elements[i] = Int(in_shape.product())
            if Int(out_shape.product()) != num_elements[i]:
                raise Error(
                    "all_to_all: an output does not hold its input's elements"
                )
            if Int(in_shape[split_axis].value()) % group_size != 0:
                raise Error(
                    String(
                        "all_to_all: split axis of size ",
                        Int(in_shape[split_axis].value()),
                        " does not divide into ",
                        group_size,
                        " chunks",
                    )
                )
            var inner_in = 1
            comptime for d in range(split_axis, rank):
                inner_in *= Int(in_shape[d].value())
            in_inner[i] = inner_in // group_size
            var inner_out = 1
            comptime for d in range(concat_axis, rank):
                inner_out *= Int(out_shape[d].value())
            out_inner[i] = inner_out // group_size

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

        @inline(.always)
        def launch_all_to_all[
            index: Int
        ]() raises {
            imm in_ptrs,
            imm out_ptrs,
            imm num_elements,
            imm in_inner,
            imm out_inner,
            imm rank_sigs,
            imm dev_ctxs_input,
        }:
            all_to_all[ngpus=num_devices, group_size=group_size](
                in_ptrs,
                out_ptrs[index],
                num_elements[index],
                in_inner[index],
                out_inner[index],
                rank_sigs,
                dev_ctxs_input[index],
                index,
            )

        _launch_device_collective[num_devices](
            launch_all_to_all, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.allgather")
struct DistributedAllGather:
    """Registers the `mo.distributed.allgather` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        _trace_name: StaticString,
        group_size: Int = 0,
        slice_rows: Bool = False,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Distributed allgather operation implementation.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            _trace_name: Trace name for profiling.
            group_size: Number of devices per allgather group; must be at
                least 1 and must evenly divide the total number of devices
                (defaults to 0).
            slice_rows: Whether each device receives only its own contiguous
                block of `dim0 / num_devices` rows of every gathered piece
                rather than the whole piece (see `allgather_rows_slice`).
                Requires one full-world group.

        Args:
            outputs: Output tensors (one per GPU) to store gathered results.
            inputs: Input tensors (one per GPU) containing values to gather.
            signal_buffers: Device buffer values used for synchronization.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert (
            len(Sigs) == num_devices and len(Outs) == num_devices * group_size
        ), (
            "expected allgather inputs, signal buffers to have the same"
            " number of elements and outputs to have num_devices * group_size"
        )
        comptime assert group_size >= 1, "group_size must be at least 1"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices * group_size, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()

        var scratch_buffer_size_bytes = 0  # no allgather impl uses scratch
        _check_signal_buffer_size(
            Int(signal_buffers[0].shape().product()), scratch_buffer_size_bytes
        )

        # World-view input/output tensors and signals, indexed by GLOBAL
        # device rank (`outputs` is already `num_devices * group_size` -- one
        # contiguous `group_size` slice per device); `allgather` does its own
        # group-local slicing internally.
        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime assert conforms_to(type_of(outputs[0]), TileTensorable)
        comptime InputTensorType = type_of(
            TileTensor(
                rebind[ImmPointer[Scalar[dtype], ImmUntrackedOrigin]](
                    inputs[0].to_tile_tensor[dtype]().ptr
                ),
                row_major(Int(inputs[0].shape().product())),
            )
        )
        comptime OutputTensorType = type_of(
            TileTensor(
                rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                    outputs[0].to_tile_tensor[dtype]().ptr
                ),
                row_major(Int(outputs[0].shape().product())),
            )
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        var out_tensors = Array[OutputTensorType, num_devices * group_size](
            uninitialized=True
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
                .ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = TileTensor(
                rebind[ImmPointer[Scalar[dtype], ImmUntrackedOrigin]](
                    inputs[i].to_tile_tensor[dtype]().ptr
                ),
                row_major(Int(inputs[i].shape().product())),
            )
        comptime for i in range(num_devices * group_size):
            comptime assert conforms_to(type_of(outputs[i]), TileTensorable)
            out_tensors[i] = TileTensor(
                rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                    outputs[i].to_tile_tensor[dtype]().ptr
                ),
                row_major(Int(outputs[i].shape().product())),
            )

        comptime if slice_rows:
            comptime assert (
                group_size == num_devices
            ), "slice_rows requires one full-world allgather group"
            comptime assert rank >= 2, "slice_rows needs a row axis to slice"
            var rows = Int(inputs[0].shape()[0].value())
            var row_elems = 1
            comptime for dim in range(1, rank):
                row_elems *= Int(inputs[0].shape()[dim].value())
            if row_elems == 0:
                if rows % num_devices != 0:
                    raise Error(
                        "allgather_rows_slice requires the row count to be a"
                        " multiple of the device count"
                    )
                comptime for i in range(num_devices):
                    if (
                        Int(inputs[i].shape()[0].value()) != rows
                        or Int(inputs[i].shape().product()) != 0
                    ):
                        raise Error(
                            "allgather_rows_slice requires empty shards with"
                            " the same row count on every device"
                        )
                comptime for i in range(num_devices * group_size):
                    if Int(outputs[i].shape().product()) != 0:
                        raise Error(
                            "allgather_rows_slice: an output buffer is not"
                            " one row block"
                        )
                return
            # `group_size == num_devices` here, which the array length can't
            # see through.
            var slice_out_tensors = Array[
                OutputTensorType, num_devices * num_devices
            ](
                fill_with_unrolled=lambda [i: Int]() -> OutputTensorType: (
                    out_tensors[i]
                )
            )

            @inline(.always)
            def launch_rows_slice[
                index: Int
            ]() raises {
                imm in_tensors,
                imm slice_out_tensors,
                imm rank_sigs,
                imm dev_ctxs_input,
                imm row_elems,
            }:
                allgather_rows_slice[ngpus=num_devices](
                    in_tensors,
                    slice_out_tensors,
                    rank_sigs,
                    row_elems,
                    dev_ctxs_input[index],
                    index,
                )

            _launch_device_collective[num_devices](
                launch_rows_slice, dev_ctxs_input.copy()
            )
        else:

            @inline(.always)
            def launch_allgather[
                index: Int
            ]() raises {
                imm in_tensors,
                imm out_tensors,
                imm rank_sigs,
                imm dev_ctxs_input,
            }:
                allgather[ngpus=num_devices, group_size=group_size](
                    in_tensors,
                    out_tensors,
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )

            _launch_device_collective[num_devices](
                launch_allgather, dev_ctxs_input.copy()
            )


@extensibility.register("mo.distributed.broadcast")
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
        Outs: TypeList[Trait=Output & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        root: Int,
        _trace_name: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        input: Some[Input & Tensor],
        signal_buffers: TensorTuple[*Sigs],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Execute distributed broadcast operation.

        Parameters:
            Outs: The element types of `outputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            root: Index of the root GPU (source of data).
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
        comptime num_devices = len(Outs)
        comptime dtype = type_of(input).dtype
        comptime rank = type_of(input).rank
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert (
            len(Sigs) == num_devices
        ), "expected 1 signal buffer per device"
        _check_tensor_tuple[num_devices, dtype, rank, *Outs]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        comptime assert (
            root >= 0 and root < num_devices
        ), "root GPU index must be in range [0, ngpus)"

        # 2-stage broadcast stages 1/ngpus of input into each signal buffer payload.
        # 1-stage broadcast doesn't use payload at all (direct P2P from root).
        # Use 2-stage requirement as upper bound.
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](Int(input.shape().product()))
        _check_signal_buffer_size(
            Int(signal_buffers[0].shape().product()), scratch_buffer_size_bytes
        )

        var in_buf = input.to_tile_tensor[dtype]()

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

        @inline(.always)
        def launch_broadcast[
            index: Int
        ]() raises {
            imm in_buf,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm outputs,
        }:
            comptime assert conforms_to(type_of(outputs[index]), TileTensorable)
            var out_buf = TileTensor(
                ptr=outputs[index].to_tile_tensor[dtype]().ptr,
                layout=in_buf.layout,
            )
            broadcast[num_devices](
                in_buf,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
                root,
                rank=index,
            )

        _launch_device_collective[num_devices](
            launch_broadcast, dev_ctxs_input.copy()
        )


@extensibility.register("mo.distributed.scatter")
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
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        root: Int,
        _trace_name: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        comptime ngpus = len(Sigs)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert (
            root >= 0 and root < ngpus
        ), "root GPU index must be in range [0, ngpus)"
        comptime assert len(Ins) == ngpus, (
            "expected scatter inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[ngpus, dtype, rank, *Ins]()
        _check_tensor_tuple[len(Outs), dtype, rank, *Outs]()
        _check_tensor_tuple[ngpus, .uint8, 1, *Sigs]()

        # Scatter uses signal buffers for barriers only (no payload staging),
        # so payload_size=0. This still validates the buffer holds a Signal.
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            Int(signal_buffers[0].shape().product()), scratch_buffer_size_bytes
        )

        # Inputs can have different static shapes, so use make_dynamic to
        # produce a homogeneous fully-dynamic TileTensor type for Array.
        comptime InputTile = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]
        var in_tensors = Array[InputTile, ngpus](uninitialized=True)

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

        var rank_sigs = Array[_, ngpus](fill_with_unrolled=rank_sig)

        comptime for i in range(ngpus):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTile](
                inputs[i].to_tile_tensor().make_dynamic[.int64]().as_imm()
            )

        @inline(.always)
        def launch_scatter[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm outputs,
        }:
            comptime assert conforms_to(type_of(outputs[index]), TileTensorable)
            var out_buf = outputs[index].to_tile_tensor[dtype]()
            scatter[ngpus=ngpus, dp_size=ngpus](
                in_tensors,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[ngpus](launch_scatter, dev_ctxs_input.copy())


@extensibility.register(
    "mo.composite.distributed.allreduce_add_rms_norm_quant_fp8"
)
struct DistributedAllReduceAddRMSNormQuantFP8:
    """Registers the `mo.composite.distributed.allreduce_add_rms_norm_quant_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        OutScales: TypeList[Trait=Output & Tensor, ...],
        OutResiduals: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Residuals: TypeList[Trait=Input & Tensor, ...],
        Gammas: TypeList[Trait=Input & Tensor, ...],
        ScalesUb: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
        # Graph compilers whose copy of the op predates the attribute fuse
        # only multiply-first norms.
        multiply_before_cast: Bool = True,
    ](
        outputs: TensorTuple[*Outs],
        outputs_scales: TensorTuple[*OutScales],
        outputs_residual: TensorTuple[*OutResiduals],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        residuals: TensorTuple[*Residuals],
        gammas: TensorTuple[*Gammas],
        scales_ub: TensorTuple[*ScalesUb],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime output_type = Outs[0].dtype
        comptime scales_type = OutScales[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, output_type, rank, *Outs]()
        _check_tensor_tuple[num_devices, scales_type, rank, *OutScales]()
        _check_tensor_tuple[num_devices, dtype, rank, *OutResiduals]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, rank, *Residuals]()
        _check_tensor_tuple[num_devices, dtype, 1, *Gammas]()
        comptime for i in range(len(ScalesUb)):
            comptime assert (
                ScalesUb[i].dtype == DType.float32
            ), "unexpected scales_ub dtype"

        # The kernel picks a path that fits the scratch behind the Signal
        # header, so only the header itself is required here.
        var signal_bytes = Int(signal_buffers[0].shape().product())
        _check_signal_buffer_size(signal_bytes, 0)
        var scratch_bytes = signal_bytes - size_of[Signal]()

        # Filter the dev_ctxs_list to have only the GPU devices.
        # The kernel also takes CPU operands (`scales_ub` are rank-0 CPU
        # scalars), so CPU devices must be removed.
        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        # Marshal input tensors into TileTensors.
        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[dtype]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)

        # Marshal signal buffers.

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

        @inline(.always)
        def launch_fused_allreduce[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs,
            imm gammas,
            imm scales_ub,
            imm outputs,
            imm outputs_scales,
            imm outputs_residual,
            imm residuals,
            imm scratch_bytes,
        }:
            # Marshal per-device outputs and residual as TileTensors.
            comptime assert conforms_to(type_of(outputs[index]), TileTensorable)
            comptime assert conforms_to(
                type_of(outputs_scales[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_residual[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(residuals[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gammas[index]), TileTensorable)
            comptime assert conforms_to(
                type_of(scales_ub[index]), TileTensorable
            )
            var out_buf = outputs[index].to_tile_tensor[output_type]()
            var out_scales_buf = outputs_scales[index].to_tile_tensor[
                scales_type
            ]()
            var out_residual_buf = outputs_residual[index].to_tile_tensor[
                dtype
            ]()
            var residual_buf = residuals[index].to_tile_tensor[dtype]().as_imm()
            var gamma_tensor = gammas[index].to_tile_tensor[dtype]()

            # TODO: Add a new struct like `VariadicInputScalar`` to
            # represent instead of manually loading the values in the
            # kernel code.
            var scale_ub = (
                scales_ub[index].to_tile_tensor[DType.float32]().ptr[]
            )

            # The kernel waits for the producing GEMM before it signals its
            # peers, and the next GEMM waits for this kernel to retire, so
            # releasing that GEMM at entry lets it prefetch weights meanwhile.
            allreduce_residual_rmsnorm[
                multiply_before_cast=multiply_before_cast
            ](
                in_tensors,
                residual_buf,
                out_buf,
                out_residual_buf,
                gamma_tensor,
                epsilon,
                weight_offset.cast[dtype](),
                scale_ub,
                out_scales_buf,
                rank_sigs,
                dev_ctxs[index],
                my_rank=index,
                scratch_bytes=scratch_bytes,
            )

        _launch_device_collective[num_devices](
            launch_fused_allreduce, dev_ctxs.copy()
        )


@extensibility.register("mo.composite.distributed.allreduce_rms_norm")
struct DistributedAllReduceRMSNorm:
    """Registers the `mo.composite.distributed.allreduce_rms_norm` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Normed: TypeList[Trait=Output & Tensor, ...],
        Summed: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Gammas: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
        multiply_before_cast: Bool = False,
    ](
        outputs_normed: TensorTuple[*Normed],
        outputs_summed: TensorTuple[*Summed],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        gammas: TensorTuple[*Gammas],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Fused allreduce sum + RMSNorm (bf16 in/out, no quantization).

        Sum-reduces `inputs` (one `[rows, cols]` tensor per device) across
        devices and RMSNorm-normalizes the exact f32 peer sum in the same
        launch, writing the normed rows to `outputs_normed`. Above the fuse
        threshold, on non-AMD targets and without P2P access, the dispatch
        falls back to a standalone allreduce into `outputs_summed` plus a
        separate norm; only that arm writes `outputs_summed`.

        Parameters:
            Normed: The element types of `outputs_normed` (inferred).
            Summed: The element types of `outputs_summed` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            Gammas: The element types of `gammas` (inferred).
            _trace_name: Trace name for profiling.
            epsilon: RMSNorm epsilon, shared by every device.
            weight_offset: Gamma offset, shared by every device.
            multiply_before_cast: The norm's gamma cast order, matching
                `ops.rms_norm`.

        Args:
            outputs_normed: Per-device normed output (same shape as `inputs`).
            outputs_summed: Per-device raw reduced sum, written by the
                two-launch arm only.
            inputs: Per-device inputs to reduce.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights (in_dtype, length cols).
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - The fused arm requires P2P access between the participating
              GPUs; without it the op runs the standalone allreduce + norm,
              whose allreduce falls back to staged peer copies.
            - The kernel stages the 2-stage path's scratch behind each
              `Signal` header; a short buffer routes to the 1-stage kernel.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Normed]()
        _check_tensor_tuple[num_devices, dtype, rank, *Summed]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, 1, *Gammas]()

        # The kernel picks a path that fits the scratch behind the Signal
        # header, so only the header itself is required here.
        var signal_bytes = Int(signal_buffers[0].shape().product())
        _check_signal_buffer_size(signal_bytes, 0)
        var scratch_bytes = signal_bytes - size_of[Signal]()

        # Marshal input tensors into TileTensors.
        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        comptime InputTensorType = TileTensor[
            dtype,
            type_of(inputs[0].to_tile_tensor[dtype]()).LayoutType,
            ImmUntrackedOrigin,
        ]
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)

        # Marshal signal buffers.

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

        @inline(.always)
        def launch_fused_allreduce_norm[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm gammas,
            imm outputs_normed,
            imm outputs_summed,
            imm scratch_bytes,
        }:
            comptime assert conforms_to(
                type_of(outputs_normed[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_summed[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gammas[index]), TileTensorable)
            var normed_buf = outputs_normed[index].to_tile_tensor[dtype]()
            var sum_buf = outputs_summed[index].to_tile_tensor[dtype]()
            var gamma_tensor = gammas[index].to_tile_tensor[dtype]()

            # Two-launch fallback (above the fuse threshold, where the
            # fabric-saturated plain allreduce wins): allreduce into
            # `sum_buf`, then `rms_norm_gpu` into `normed_buf`, matching the
            # unfused chain including the attribute's cast order.
            @inline(.always)
            def two_launch() raises {imm}:
                comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
                    logger.info("Executing: Vendor CCL")

                    # _get_global_comms has a check-then-create race: two
                    # threads seeing null simultaneously would both call
                    # ncclCommInitAll and leak one set of communicators.
                    # Only device 0 initializes; others spin-wait.
                    comptime if index == 0:
                        vendor_ccl.init_comms(num_devices)
                    else:
                        vendor_ccl.wait_for_comms(num_devices)

                    # Writes this device's reduced sum into `sum_buf`. It
                    # runs on the device, so it copies the buffer.
                    @inline(.always)
                    def output_lambda[
                        _dtype: DType,
                        _width: SIMDLength,
                        *,
                        alignment: Int,
                    ](coords: Coord, val: SIMD[_dtype, _width]) {
                        var sum_buf
                    } -> None:
                        sum_buf.store[alignment=alignment](
                            coords, rebind[SIMD[dtype, _width]](val)
                        )

                    vendor_ccl.allreduce[ngpus=num_devices](
                        in_tensors,
                        row_major(sum_buf.layout.shape_coord()),
                        output_lambda,
                        rank_sigs,
                        dev_ctxs_input[index],
                    )
                else:
                    # The standalone allreduce's 2-stage path stages
                    # payload/num_devices bytes behind each Signal header with
                    # no internal check; the plain op's registration enforces
                    # that sizing, so match it here. The fused arm needs no
                    # such check: it picks a kernel that fits the scratch.
                    # This runs inside the concurrent per-device launch, so it
                    # must stay rank-identical (it reads only the shared shape
                    # and signal-buffer size): a raise on one rank alone would
                    # strand its peers at the barrier.
                    _check_signal_buffer_size(
                        size_of[Signal]() + scratch_bytes,
                        _partitioned_scratch_requirement[num_devices, dtype](
                            sum_buf.num_elements()
                        ),
                    )
                    allreduce[ngpus=num_devices](
                        in_tensors,
                        sum_buf,
                        rank_sigs,
                        dev_ctxs_input[index],
                        my_rank=index,
                    )

                # The `{var}` copies are REQUIRED: run by the `rms_norm_gpu`
                # device kernel, a local `var` (`sum_buf`/`normed_buf`)
                # captured by reference is not carried to the device -> the
                # closure reads a garbage host-stack pointer and corrupts
                # device memory.
                @inline(.always)
                def norm_input_fn[
                    width: Int
                ](coords: Coord) {var sum_buf} -> SIMD[dtype, width]:
                    return sum_buf.load[
                        width=width, alignment=align_of[dtype]()
                    ](coords)

                @inline(.always)
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) {
                    var normed_buf
                } -> None:
                    normed_buf.store[alignment=alignment](coords, val)

                rms_norm_gpu[
                    rank,
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=multiply_before_cast,
                ](
                    norm_input_fn,
                    norm_output_fn,
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    dev_ctxs_input[index],
                )

            # The vendor CCL build always reduces through the vendor
            # allreduce, as `mo.distributed.allreduce.sum` does.
            comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
                two_launch()
                return

            _dispatch_ar_norm[multiply_before_cast=multiply_before_cast](
                in_tensors,
                normed_buf,
                sum_buf,
                gamma_tensor,
                epsilon,
                weight_offset.cast[dtype](),
                rank_sigs,
                dev_ctxs_input[index],
                two_launch,
                my_rank=index,
                scratch_bytes=scratch_bytes,
            )

        _launch_device_collective[num_devices](
            launch_fused_allreduce_norm, dev_ctxs_input.copy()
        )


@extensibility.register("mo.composite.distributed.reduce_scatter_rms_norm")
struct DistributedReduceScatterRMSNorm:
    """Registers the `mo.composite.distributed.reduce_scatter_rms_norm` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Normed: TypeList[Trait=Output & Tensor, ...],
        Sums: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Gammas: TypeList[Trait=Input & Tensor, ...],
        Residuals: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
        group_size: Int = 0,
        has_residual: Bool = False,
        quant_bits: Int = 0,
    ](
        outputs_normed: TensorTuple[*Normed],
        outputs_sum: TensorTuple[*Sums],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        gammas: TensorTuple[*Gammas],
        residuals: TensorTuple[*Residuals],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Fused reduce-scatter sum + RMSNorm + residual add (bf16 out).

        Reduce-scatters `inputs` (one `[rows, cols]` tensor per device) along
        rows, adds `residuals`, and RMSNorm-normalizes each owned shard in the
        same launch, writing the normed shard to `outputs_normed` and the summed
        shard (the residual stream) to `outputs_sum`.

        Under `has_residual`, `residuals` carries the TP-replicated residual
        stream. Each device adds only its own row shard of it, which is why
        such callers must NOT pre-add it on the group leader: the
        reduce-scatter sums across ranks, so a leader-side add lands once for
        the whole group and this per-rank add reproduces it exactly -- but only
        because the residual is bit-identical on every rank of the group.
        Folding it here deletes a full-width elementwise add that ran on the
        group leader alone.

        Without `has_residual` the operands are still passed (the op's variadic
        groups must match in size) but never read, and both arms are exactly
        the reduce-scatter + norm this op was before the fold existed.

        Parameters:
            Normed: The element types of `outputs_normed` (inferred).
            Sums: The element types of `outputs_sum` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            Gammas: The element types of `gammas` (inferred).
            Residuals: The element types of `residuals` (inferred).
            _trace_name: Trace name for profiling.
            epsilon: RMSNorm epsilon, shared by every device.
            weight_offset: Gamma offset, shared by every device.
            group_size: Number of contiguous devices per independent
                reduce-scatter group; must be at least 2 and must evenly divide
                the total number of devices. Equal to `num_devices` for a
                full-world collective. The builder always sets it; the `0`
                attribute default is not a usable value.
            has_residual: Fold `residuals` into the reduce-scatter sum. Off
                leaves both arms byte-for-byte the plain reduce-scatter + norm.
            quant_bits: 4 or 8 to send the two-launch arm's peer traffic
                quantized (INT4 or MXFP8, lossy) when the whole world reduces
                together, the columns are a multiple of 32 and the signal
                buffers hold the scratch; 0 keeps it exact.

        Args:
            outputs_normed: Per-device normed output shards.
            outputs_sum: Per-device reduce-scatter sum shards (residual stream).
            inputs: Per-device input tensors to reduce and scatter.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights (in_dtype, length cols).
            residuals: Per-device residual stream, same shape as `inputs` and
                bit-identical across each group. Read only under
                `has_residual`.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo).
            - Requires P2P, and identical tensor shapes within each group.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected reduce_scatter_rms_norm inputs and signal buffers to have"
            " the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Normed]()
        _check_tensor_tuple[num_devices, dtype, rank, *Sums]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, 1, *Gammas]()
        _check_tensor_tuple[num_devices, dtype, rank, *Residuals]()
        # >= 2, unlike the plain op: the fused kernel asserts `ngpus >= 2` and
        # this composite has no 1-input folder, so a 1-device group would fail
        # deeper with an unrelated message.
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Like plain reduce-scatter, no scratch: only the Signal struct. The
        # quantized arm checks its own scratch and column alignment below and
        # falls back without them.
        var signal_bytes = Int(signal_buffers[0].shape().product())
        _check_signal_buffer_size(signal_bytes, 0)
        comptime assert quant_bits in (0, 4, 8), "quant_bits must be 0, 4 or 8"
        comptime quantize = quant_bits > 0 and group_size == num_devices
        var quant_scratch_fits = False
        comptime if quantize:
            var rs_cols = Int(inputs[0].shape()[rank - 1].value())
            var rs_rows = Int(inputs[0].shape().product()) // rs_cols
            quant_scratch_fits = (
                rs_cols % RS_QUANT_GROUP == 0
                and signal_bytes
                >= size_of[Signal]()
                + rs_quant_scratch_bytes[quant_bits, num_devices](
                    rs_rows, rs_cols
                )
            )

        # World-view input tensors and signals, indexed by GLOBAL device rank;
        # `reducescatter`/`_dispatch_rs_norm` do their own group-local slicing
        # internally.
        comptime InputTensorType = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]
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
                inputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )
        _check_uniform_cols[op="reduce_scatter_rms_norm"](inputs)

        # `reducescatter`'s output and residual are world views too (every
        # device's own shard, indexed by global rank). Every slot has to name
        # a real buffer, not just this device's: a grouped reduce-scatter may
        # hand part of a shard to the paired group, whose GPUs then finish it
        # and write it directly, folding that group's residual as they go.
        # Nothing in either depends on which device is launching.
        comptime SumTensorType = TileTensor[
            mut=True,
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            MutUntrackedOrigin,
        ]
        comptime ResPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        var world_sum_bufs = Array[SumTensorType, num_devices](
            uninitialized=True
        )
        var world_res_ptrs = Array[ResPtrType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(outputs_sum[i]), TileTensorable)
            comptime assert conforms_to(type_of(residuals[i]), TileTensorable)
            world_sum_bufs[i] = rebind[SumTensorType](
                outputs_sum[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )
            # The op's variadic groups match in size whether or not the
            # residual is read, so this is always a real tensor.
            world_res_ptrs[i] = rebind[ResPtrType](
                residuals[i].to_tile_tensor[dtype]().as_imm().ptr
            )

        @inline(.always)
        def launch_fused_rs_norm[
            index: Int
        ]() raises {
            imm in_tensors,
            imm quant_scratch_fits,
            imm rank_sigs,
            imm world_sum_bufs,
            imm world_res_ptrs,
            imm dev_ctxs_input,
            imm gammas,
            imm residuals,
            imm outputs_normed,
            imm outputs_sum,
        }:
            # `local_rank` (this device's rank WITHIN its group) is still
            # needed here: `sum_buf`/`normed_buf` are this device's own
            # OUTPUT, indexed by the group-local row, not the global one.
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size

            comptime assert conforms_to(
                type_of(outputs_normed[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_sum[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gammas[index]), TileTensorable)
            comptime assert conforms_to(
                type_of(residuals[index]), TileTensorable
            )
            var normed_buf = outputs_normed[index].to_tile_tensor[dtype]()
            var sum_buf = outputs_sum[index].to_tile_tensor[dtype]()
            var gamma_tensor = gammas[index].to_tile_tensor[dtype]()
            var residual_buf = rebind[InputTensorType](
                residuals[index].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )

            @inline(.always)
            def rs_exact() raises {imm}:
                comptime if has_residual:
                    # Handed to `reducescatter` as a residual rather than as
                    # an epilogue: an epilogue would rule out the relay path,
                    # which cannot run a caller's closure on a peer's behalf,
                    # whereas a residual fold it can reproduce.
                    reducescatter[
                        dtype=dtype,
                        ngpus=num_devices,
                        group_size=group_size,
                        axis=0,
                        has_residual=True,
                    ](
                        in_tensors,
                        world_sum_bufs,
                        rank_sigs,
                        dev_ctxs_input[index],
                        my_rank=index,
                        residuals=Optional(world_res_ptrs.copy()),
                    )
                else:
                    reducescatter[
                        dtype=dtype,
                        ngpus=num_devices,
                        group_size=group_size,
                        axis=0,
                    ](
                        in_tensors,
                        world_sum_bufs,
                        rank_sigs,
                        dev_ctxs_input[index],
                        my_rank=index,
                    )

            @inline(.always)
            def two_launch() raises {imm}:
                comptime if quantize:
                    if quant_scratch_fits:
                        var rs_cols = Int(sum_buf.dim[rank - 1]())
                        reducescatter_quant[
                            dtype, num_devices, quant_bits, has_residual
                        ](
                            in_tensors[index].ptr.as_unsafe_any_origin(),
                            sum_buf.ptr.as_unsafe_any_origin(),
                            world_res_ptrs[index],
                            in_tensors[index].num_elements() // rs_cols,
                            rs_cols,
                            rank_sigs,
                            index,
                            dev_ctxs_input[index],
                        )
                    else:
                        rs_exact()
                else:
                    rs_exact()

                # The `{var}` copies are REQUIRED: run by the `rms_norm_gpu`
                # device kernel, a local `var` (`sum_buf`/`normed_buf`)
                # captured by reference is not carried to the device -> the
                # closure reads a garbage host-stack pointer and corrupts
                # device memory.
                @inline(.always)
                def norm_input_fn[
                    width: Int
                ](coords: Coord) {var sum_buf} -> SIMD[dtype, width]:
                    return sum_buf.load[
                        width=width, alignment=align_of[dtype]()
                    ](coords)

                @inline(.always)
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) {
                    var normed_buf
                } -> None:
                    normed_buf.store[alignment=alignment](coords, val)

                rms_norm_gpu[
                    rank,
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=True,
                ](
                    norm_input_fn,
                    norm_output_fn,
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    dev_ctxs_input[index],
                )

            # The unfused `reducescatter` this fusion replaced already used PDL,
            # so leaving it off here would regress the path.
            # `residual`'s disengaged default is a distinct type, so it cannot
            # be passed unconditionally -- hence the two call sites.
            comptime if has_residual:
                _dispatch_rs_norm[
                    has_residual=True,
                    group_size=group_size,
                ](
                    in_tensors,
                    normed_buf,
                    sum_buf,
                    gamma_tensor,
                    epsilon,
                    weight_offset.cast[dtype](),
                    rank_sigs,
                    dev_ctxs_input[index],
                    two_launch,
                    my_rank=index,
                    residual=residual_buf,
                )
            else:
                _dispatch_rs_norm[
                    has_residual=False,
                    group_size=group_size,
                ](
                    in_tensors,
                    normed_buf,
                    sum_buf,
                    gamma_tensor,
                    epsilon,
                    weight_offset.cast[dtype](),
                    rank_sigs,
                    dev_ctxs_input[index],
                    two_launch,
                    my_rank=index,
                )

        _launch_device_collective[num_devices](
            launch_fused_rs_norm, dev_ctxs_input.copy()
        )


@extensibility.register("mo.composite.distributed.allgather_rms_norm")
struct DistributedAllGatherRMSNorm:
    """Registers the `mo.composite.distributed.allgather_rms_norm` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        Normed: TypeList[Trait=Output & Tensor, ...],
        Residuals: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Gammas: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
        group_size: Int = 0,
        gather_axis: Int = 0,
        round_offset_gamma: Bool = False,
    ](
        outputs_normed: TensorTuple[*Normed],
        outputs_residual: TensorTuple[*Residuals],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        gammas: TensorTuple[*Gammas],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Fused all-gather + RMSNorm (bf16 in/out, no quantization).

        All-gathers `inputs` (per-device `[shard_i, cols]` row-shards) into the
        full replicated `[rows, cols]` stream and RMSNorms every gathered row in
        one launch: normed to `outputs_normed`, gathered residual to
        `outputs_residual` (both full and replicated).

        Parameters:
            Normed: The element types of `outputs_normed` (inferred).
            Residuals: The element types of `outputs_residual` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            Gammas: The element types of `gammas` (inferred).
            _trace_name: Trace name for profiling.
            epsilon: RMSNorm epsilon, shared by every device.
            weight_offset: Gamma offset, shared by every device.
            group_size: Number of contiguous devices per independent all-gather
                group; must be at least 2 and must evenly divide the total
                number of devices. Equal to `num_devices` for a full-world
                collective. The builder always sets it; the `0` attribute
                default is not a usable value.
            gather_axis: 0 to gather row shards, 1 column shards.
            round_offset_gamma: Whether `gamma + weight_offset` is rounded to
                `dtype` before the f32 multiply.

        Args:
            outputs_normed: Per-device normed output (the group's gathered
                `[rows, cols]`, replicated within the group).
            outputs_residual: Per-device gathered residual (same shape as
                `outputs_normed`; the residual stream).
            inputs: Per-device input row-shards to gather.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights (in_dtype, length cols).
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo).
            - Requires P2P; within a group, shard shapes may differ only in the
              gathered axis.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allgather_rms_norm inputs and signal buffers to have the"
            " same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Normed]()
        _check_tensor_tuple[num_devices, dtype, rank, *Residuals]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, 1, *Gammas]()
        # >= 2, unlike the plain op: the fused kernel asserts `ngpus >= 2` and
        # this composite has no 1-input folder, so a 1-device group would fail
        # deeper with an unrelated message.
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"

        # Like plain all-gather, no scratch: only the Signal struct.
        _check_signal_buffer_size(Int(signal_buffers[0].shape().product()), 0)

        # World-view input tensors and signals, indexed by GLOBAL device
        # rank. Groups may carry different static shapes, so `make_dynamic`
        # erases the extents: a `rebind` to a type still carrying device 0's
        # would fail to elaborate.
        comptime InputTensorType = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]
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
                inputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )
        _check_uniform_cols[op="allgather_rms_norm"](inputs)

        # The two-launch fallback below gathers rows and folds the offset in
        # f32, so these modes always fuse.
        comptime if gather_axis == 1 or round_offset_gamma:

            @inline(.always)
            def launch_fused[
                index: Int
            ]() raises {
                imm in_tensors,
                imm rank_sigs,
                imm dev_ctxs_input,
                imm gammas,
                imm outputs_normed,
                imm outputs_residual,
            }:
                comptime assert conforms_to(
                    type_of(outputs_normed[index]), TileTensorable
                )
                comptime assert conforms_to(
                    type_of(outputs_residual[index]), TileTensorable
                )
                comptime assert conforms_to(
                    type_of(gammas[index]), TileTensorable
                )
                # The same PDL level the row-gather dispatch takes.
                comptime pdl_level = (
                    PDLLevel.ON if num_devices == 2 else PDLLevel.OFF
                )
                allgather_rmsnorm[
                    group_size=group_size,
                    pdl_level=pdl_level,
                    gather_cols=gather_axis == 1,
                    round_offset_gamma=round_offset_gamma,
                ](
                    in_tensors,
                    outputs_normed[index].to_tile_tensor[dtype](),
                    outputs_residual[index].to_tile_tensor[dtype](),
                    gammas[index].to_tile_tensor[dtype](),
                    epsilon,
                    weight_offset.cast[dtype](),
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )

            _launch_device_collective[num_devices](
                launch_fused, dev_ctxs_input.copy()
            )
            return

        # `allgather`'s world-view output array holds every device's own
        # `group_size` output windows. Every slot has to name a real window,
        # not just this device's: a grouped allgather may route part of a
        # shard through the paired group, whose GPUs then write these buffers
        # directly. Each device windows its own output by ITS group's ragged
        # row counts. Built once here; nothing in it depends on which device
        # is launching, and the column count is uniform across the world.
        comptime assert conforms_to(
            type_of(outputs_residual[0]), TileTensorable
        )
        var world_cols = Int(
            outputs_residual[0].to_tile_tensor[dtype]().dim[rank - 1]()
        )
        comptime OutViewType = type_of(
            TileTensor(
                rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                    outputs_residual[0].to_tile_tensor[dtype]().ptr
                ),
                row_major(world_cols, world_cols),
            )
        )
        var world_out_views = Array[OutViewType, num_devices * group_size](
            uninitialized=True
        )
        comptime for d in range(num_devices):
            comptime d_group_start = ualign_down(d, group_size)
            comptime assert conforms_to(
                type_of(outputs_residual[d]), TileTensorable
            )
            var d_base = rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                outputs_residual[d].to_tile_tensor[dtype]().ptr
            )
            var row_off = 0
            comptime for i in range(group_size):
                var len_i = (
                    Int(in_tensors[d_group_start + i].num_elements())
                    // world_cols
                )
                world_out_views[d * group_size + i] = TileTensor(
                    d_base.unsafe_offset(row_off * world_cols),
                    row_major(len_i, world_cols),
                )
                row_off += len_i

        @inline(.always)
        def launch_fused_ag_norm[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm world_out_views,
            imm dev_ctxs_input,
            imm gammas,
            imm outputs_normed,
            imm outputs_residual,
        }:
            # `local_rank`/`group_start` are still needed here: `two_launch`
            # windows its OWN world-view `allgather` call by group, and the
            # row-offset math below reads THIS DEVICE'S GROUP's inputs.
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size

            comptime assert conforms_to(
                type_of(outputs_normed[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_residual[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gammas[index]), TileTensorable)
            var normed_buf = outputs_normed[index].to_tile_tensor[dtype]()
            var sum_buf = outputs_residual[index].to_tile_tensor[dtype]()
            var gamma_tensor = gammas[index].to_tile_tensor[dtype]()

            # Two-launch fallback (above the fuse threshold, where the
            # fabric-saturated standalone gather wins): all-gather into `sum_buf`,
            # then `rms_norm_gpu` into `normed_buf`. `sum_buf` is the residual on
            # both branches. mbc=True.
            @inline(.always)
            def two_launch() raises {imm}:
                # Each shard gathers into its contiguous row-range of
                # `sum_buf` (natural concat order), so the norm below runs
                # over the whole tensor.
                allgather[
                    dtype=dtype, ngpus=num_devices, group_size=group_size
                ](
                    in_tensors,
                    world_out_views,
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )

                # The `{var}` copies are REQUIRED: captured by reference, the
                # local `var` (`sum_buf`/`normed_buf`) reaches the
                # `rms_norm_gpu` device kernel as a garbage host-stack pointer
                # and corrupts memory.
                @inline(.always)
                def norm_input_fn[
                    width: Int
                ](coords: Coord) {var sum_buf} -> SIMD[dtype, width]:
                    return sum_buf.load[
                        width=width, alignment=align_of[dtype]()
                    ](coords)

                @inline(.always)
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) {
                    var normed_buf
                } -> None:
                    normed_buf.store[alignment=alignment](coords, val)

                rms_norm_gpu[
                    rank,
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=True,
                ](
                    norm_input_fn,
                    norm_output_fn,
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    dev_ctxs_input[index],
                )

            _dispatch_ag_norm[group_size=group_size](
                in_tensors,
                normed_buf,
                sum_buf,
                gamma_tensor,
                epsilon,
                weight_offset.cast[dtype](),
                rank_sigs,
                dev_ctxs_input[index],
                two_launch,
                my_rank=index,
            )

        _launch_device_collective[num_devices](
            launch_fused_ag_norm, dev_ctxs_input.copy()
        )


@extensibility.register(
    "mo.composite.distributed.allgather_gated_residual_rms_norm"
)
struct DistributedAllGatherGatedResidualRMSNorm:
    """Registers `mo.composite.distributed.allgather_gated_residual_rms_norm`.
    """

    @staticmethod
    def execute[
        Streams: TypeList[Trait=Output & Tensor, ...],
        Normed: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Residuals: TypeList[Trait=Input & Tensor, ...],
        PostWeights: TypeList[Trait=Input & Tensor, ...],
        Gates: TypeList[Trait=Input & Tensor, ...],
        NormWeights: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        post_eps: Float32,
        norm_eps: Float32,
        temperature: Float32,
        gate_eps: Float32,
        scale_branch: Bool,
    ](
        outputs_stream: TensorTuple[*Streams],
        outputs_normed: TensorTuple[*Normed],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        residuals: TensorTuple[*Residuals],
        post_weights: TensorTuple[*PostWeights],
        gates: TensorTuple[*Gates],
        norm_weights: TensorTuple[*NormWeights],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Gathers a branch's column shards and joins it to each device's stream.

        Parameters:
            Streams: The element types of `outputs_stream` (inferred).
            Normed: The element types of `outputs_normed` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            Residuals: The element types of `residuals` (inferred).
            PostWeights: The element types of `post_weights` (inferred).
            Gates: The element types of `gates` (inferred).
            NormWeights: The element types of `norm_weights` (inferred).
            _trace_name: Trace name for profiling.
            post_eps: The branch norm's epsilon.
            norm_eps: The next pre-norm's epsilon.
            temperature: The gate's temperature.
            gate_eps: The floor under the residual coefficient's root.
            scale_branch: Whether the branch is scaled by `1 + post_weight`
                before its norm.

        Args:
            outputs_stream: Per-device new residual stream `[rows, cols]`.
            outputs_normed: Per-device next pre-norm of it `[rows, cols]`.
            inputs: Per-device branch column shards `[rows, cols / N]`.
            signal_buffers: Per-device synchronization buffers.
            residuals: Per-device incoming residual stream `[rows, cols]`.
            post_weights: Per-device branch input scale weight `[cols]`.
            gates: Per-device residual gate logits `[cols]`.
            norm_weights: Per-device next pre-norm weight `[cols]`.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime gate_dtype = Gates[0].dtype
        comptime assert len(Sigs) == num_devices, (
            "expected one signal buffer per allgather_gated_residual_rms_norm"
            " input"
        )
        _check_tensor_tuple[num_devices, dtype, 2, *Ins]()
        _check_tensor_tuple[num_devices, dtype, 2, *Streams]()
        _check_tensor_tuple[num_devices, dtype, 2, *Normed]()
        _check_tensor_tuple[num_devices, dtype, 2, *Residuals]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, 1, *PostWeights]()
        _check_tensor_tuple[num_devices, gate_dtype, 1, *Gates]()
        _check_tensor_tuple[num_devices, dtype, 1, *NormWeights]()
        _check_signal_buffer_size(Int(signal_buffers[0].shape().product()), 0)

        comptime InputTensorType = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, 2].element_types,
                stride_types=DynamicCoord[.int64, 2].element_types,
            ],
            ImmUntrackedOrigin,
        ]
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        comptime for i in range(num_devices):
            comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
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
                .ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )

        var rank_sigs = Array[_, num_devices](fill_with_unrolled=rank_sig)
        _check_uniform_cols[op="allgather_gated_residual_rms_norm"](inputs)

        @inline(.always)
        def launch[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm dev_ctxs_input,
            imm residuals,
            imm post_weights,
            imm gates,
            imm norm_weights,
            imm outputs_stream,
            imm outputs_normed,
        }:
            comptime assert conforms_to(
                type_of(residuals[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(post_weights[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gates[index]), TileTensorable)
            comptime assert conforms_to(
                type_of(norm_weights[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_stream[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_normed[index]), TileTensorable
            )
            allgather_gated_residual_rmsnorm[scale_branch=scale_branch](
                in_tensors,
                residuals[index].to_tile_tensor[dtype](),
                post_weights[index].to_tile_tensor[dtype](),
                gates[index].to_tile_tensor[gate_dtype](),
                norm_weights[index].to_tile_tensor[dtype](),
                outputs_stream[index].to_tile_tensor[dtype](),
                outputs_normed[index].to_tile_tensor[dtype](),
                post_eps,
                norm_eps,
                temperature,
                gate_eps,
                rank_sigs,
                dev_ctxs_input[index],
                index,
            )

        _launch_device_collective[num_devices](launch, dev_ctxs_input.copy())


@extensibility.register(
    "mo.composite.distributed.allgather_rms_norm_quant_mxfp8"
)
struct DistributedAllGatherRMSNormQuantMXFP8:
    """Registers `mo.composite.distributed.allgather_rms_norm_quant_mxfp8`."""

    @staticmethod
    def execute[
        Normed: TypeList[Trait=Output & Tensor, ...],
        Quant: TypeList[Trait=Output & Tensor, ...],
        Scales: TypeList[Trait=Output & Tensor, ...],
        Residuals: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Gammas: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
        group_size: Int = 0,
        gather_quant_only: Bool = False,
    ](
        outputs_normed: TensorTuple[*Normed],
        outputs_quant: TensorTuple[*Quant],
        outputs_scale: TensorTuple[*Scales],
        outputs_residual: TensorTuple[*Residuals],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        gammas: TensorTuple[*Gammas],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Fused all-gather + RMSNorm that also emits an MXFP8 copy.

        `DistributedAllGatherRMSNorm` plus `outputs_quant`/`outputs_scale`. The
        quantize rides the collective's epilogue on the same bf16 that lands in
        `outputs_normed`, so it is byte-identical to a standalone quantize.

        Parameters:
            Normed: The element types of `outputs_normed` (inferred).
            Quant: The element types of `outputs_quant` (inferred).
            Scales: The element types of `outputs_scale` (inferred).
            Residuals: The element types of `outputs_residual` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            Gammas: The element types of `gammas` (inferred).
            _trace_name: Trace name for profiling.
            epsilon: RMSNorm epsilon, shared by every device.
            weight_offset: Gamma offset, shared by every device.
            group_size: Devices per independent all-gather group; see
                `DistributedAllGatherRMSNorm`.
            gather_quant_only: When every group gathers at least
                `AG_QUANT_ONLY_THRESHOLD` bytes of bf16 rows (~1024 rows), norm
                and quantize only this device's row shard and gather just the
                MXFP8 values and scales, about half the bytes of the bf16
                residual; below it the extra scales gather costs more than the
                bytes save. Each shard is normed with its owner's gamma, so the
                result matches the default only when every device in a group
                holds the same gamma. `outputs_normed` and `outputs_residual`
                then hold valid rows only for this device's shard. The fused
                path is unchanged.

        Args:
            outputs_normed: Per-device normed output (the group's gathered
                `[rows, cols]`, replicated within the group).
            outputs_quant: Per-device MXFP8 copy of `outputs_normed`.
            outputs_scale: Per-device E8M0 block scales, `[rows, cols / 32]`.
            outputs_residual: Per-device gathered residual.
            inputs: Per-device input row-shards to gather.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Targets AMD (CDNA4): the fallback calls `quantize_mx_amd` and
              the scale layout is its rank-2 one -- what
              `block_scaled_matmul_amd` takes as `a_scales`, not the SM100 SF
              atom and not the preshuffled order the `_preb` kernel needs.
            - Maximum of 8 GPUs; requires P2P.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime quant_dtype = Quant[0].dtype
        comptime scales_dtype = Scales[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allgather_rms_norm_quant_mxfp8 inputs and signal buffers"
            " to have the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Normed]()
        _check_tensor_tuple[num_devices, quant_dtype, rank, *Quant]()
        _check_tensor_tuple[num_devices, scales_dtype, rank, *Scales]()
        _check_tensor_tuple[num_devices, dtype, rank, *Residuals]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, 1, *Gammas]()
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"
        comptime assert (
            quant_dtype == .float8_e4m3fn
        ), "MXFP8 quant output must be float8_e4m3fn"
        comptime assert (
            scales_dtype == .float8_e8m0fnu
        ), "MXFP8 block scales must be float8_e8m0fnu"

        _check_signal_buffer_size(Int(signal_buffers[0].shape().product()), 0)

        # World-view input tensors and signals, indexed by GLOBAL device
        # rank; see `DistributedAllGatherRMSNorm`.
        comptime InputTensorType = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]
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
                inputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )
        _check_uniform_cols[op="allgather_rms_norm_quant_mxfp8"](inputs)

        # `allgather`'s world-view output array holds every device's own
        # `group_size` output windows. Every slot has to name a real window,
        # not just this device's: a grouped allgather may route part of a
        # shard through the paired group, whose GPUs then write these buffers
        # directly. Each device windows its own output by ITS group's ragged
        # row counts. Built once here; nothing in it depends on which device
        # is launching, and the column count is uniform across the world.
        comptime assert conforms_to(
            type_of(outputs_residual[0]), TileTensorable
        )
        var world_cols = Int(
            outputs_residual[0].to_tile_tensor[dtype]().dim[rank - 1]()
        )
        comptime OutViewType = type_of(
            TileTensor(
                rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                    outputs_residual[0].to_tile_tensor[dtype]().ptr
                ),
                row_major(world_cols, world_cols),
            )
        )
        var world_out_views = Array[OutViewType, num_devices * group_size](
            uninitialized=True
        )
        comptime for d in range(num_devices):
            comptime d_group_start = ualign_down(d, group_size)
            comptime assert conforms_to(
                type_of(outputs_residual[d]), TileTensorable
            )
            var d_base = rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                outputs_residual[d].to_tile_tensor[dtype]().ptr
            )
            var row_off = 0
            comptime for i in range(group_size):
                var len_i = (
                    Int(in_tensors[d_group_start + i].num_elements())
                    // world_cols
                )
                world_out_views[d * group_size + i] = TileTensor(
                    d_base.unsafe_offset(row_off * world_cols),
                    row_major(len_i, world_cols),
                )
                row_off += len_i

        @inline(.always)
        def launch_fused_ag_norm_quant[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm world_out_views,
            imm dev_ctxs_input,
            imm gammas,
            imm outputs_normed,
            imm outputs_quant,
            imm outputs_scale,
            imm outputs_residual,
        }:
            # `local_rank`/`group_start`: see `DistributedAllGatherRMSNorm`.
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size

            comptime assert conforms_to(
                type_of(outputs_normed[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_quant[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_scale[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_residual[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gammas[index]), TileTensorable)
            var normed_buf = outputs_normed[index].to_tile_tensor[dtype]()
            var quant_buf = outputs_quant[index].to_tile_tensor[quant_dtype]()
            var scale_buf = outputs_scale[index].to_tile_tensor[scales_dtype]()
            var sum_buf = outputs_residual[index].to_tile_tensor[dtype]()
            var gamma_tensor = gammas[index].to_tile_tensor[dtype]()
            var cols_rt = Int(sum_buf.dim[rank - 1]())

            # Both write paths index the quant/scale destinations by computed
            # offset with no bound on the destination's own extents: the fused
            # epilogue's `store` does no bounds check of its own, and
            # `quantize_mx_amd` derives its geometry from the INPUT. So an
            # undersized scale tensor writes into the next row -- self-consistent
            # garbage a byte-compare oracle cannot see -- and past the allocation
            # on the last one. Mirrors the `normed_out`/`sum_out` guards in
            # `comm/allgather_rmsnorm.mojo`, which exist because sizing an output
            # from the whole world instead of the TP group is the natural mistake
            # here; these two only reach the kernel through a closure.
            var rows_rt = normed_buf.num_elements() // cols_rt
            var quant_cols = Int(quant_buf.dim[rank - 1]())
            if (
                quant_cols != cols_rt
                or quant_buf.num_elements() != rows_rt * cols_rt
            ):
                raise Error(
                    String(
                        "allgather_rms_norm_quant_mxfp8: outputs_quant is ",
                        quant_buf.num_elements() // quant_cols,
                        " x ",
                        quant_cols,
                        ", expected ",
                        rows_rt,
                        " x ",
                        cols_rt,
                        " (as outputs_normed)",
                    )
                )
            var scale_cols = Int(scale_buf.dim[rank - 1]())
            if (
                scale_cols * MXFP8_SF_VECTOR_SIZE != cols_rt
                or scale_buf.num_elements() != rows_rt * scale_cols
            ):
                raise Error(
                    String(
                        "allgather_rms_norm_quant_mxfp8: outputs_scale is ",
                        scale_buf.num_elements() // scale_cols,
                        " x ",
                        scale_cols,
                        ", expected ",
                        rows_rt,
                        " x ",
                        cols_rt // MXFP8_SF_VECTOR_SIZE,
                        " (normed cols / ",
                        MXFP8_SF_VECTOR_SIZE,
                        ")",
                    )
                )

            # The `{var}` copies are REQUIRED: a local this closure captures by
            # reference reaches the device as a host-stack pointer and faults.
            @inline(.always)
            def mx_epilogue[
                width: Int
            ](row: Int, col: Int, val: SIMD[dtype, width]) {
                var quant_buf, var scale_buf, var cols_rt
            }:
                var quantized: SIMD[quant_dtype, width]
                var e8m0: Scalar[scales_dtype]
                quantized, e8m0 = quantize_mxfp8_lane_group[
                    quant_dtype,
                    scales_dtype,
                    SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE,
                ](val)
                # Quantize on every lane -- the block max is a cross-lane
                # reduction. Only the in-range lanes store.
                if col < cols_rt:
                    quant_buf.store[alignment=align_of[quant_dtype]()](
                        Coord(row, col), quantized
                    )
                    if col % MXFP8_SF_VECTOR_SIZE == 0:
                        scale_buf[row, col // MXFP8_SF_VECTOR_SIZE] = e8m0

            # Above the fuse threshold. Owes the same outputs as the fused
            # path -- skipping the quantize leaves `outputs_quant` stale.
            @inline(.always)
            def two_launch_with_quant() raises {imm}:
                allgather[
                    dtype=dtype, ngpus=num_devices, group_size=group_size
                ](
                    in_tensors,
                    world_out_views,
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )

                @inline(.always)
                def norm_input_fn[
                    width: Int
                ](coords: Coord) {var sum_buf} -> SIMD[dtype, width]:
                    return sum_buf.load[
                        width=width, alignment=align_of[dtype]()
                    ](coords)

                @inline(.always)
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) {
                    var normed_buf
                } -> None:
                    normed_buf.store[alignment=alignment](coords, val)

                rms_norm_gpu[
                    rank,
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=True,
                ](
                    norm_input_fn,
                    norm_output_fn,
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    dev_ctxs_input[index],
                )

                quantize_mx_amd(
                    dev_ctxs_input[index], quant_buf, scale_buf, normed_buf
                )

            @inline(.always)
            def two_launch_quant_only() raises {imm}:
                comptime QuantPtrType = MutPointer[
                    Scalar[quant_dtype], MutUntrackedOrigin
                ]
                comptime ScalePtrType = MutPointer[
                    Scalar[scales_dtype], MutUntrackedOrigin
                ]
                comptime QuantViewType = type_of(
                    TileTensor(
                        rebind[QuantPtrType](quant_buf.ptr),
                        row_major(cols_rt, cols_rt),
                    )
                )
                comptime ScaleViewType = type_of(
                    TileTensor(
                        rebind[ScalePtrType](scale_buf.ptr),
                        row_major(cols_rt, scale_cols),
                    )
                )
                var quant_views = Array[
                    QuantViewType, num_devices * group_size
                ](uninitialized=True)
                var scale_views = Array[
                    ScaleViewType, num_devices * group_size
                ](uninitialized=True)
                var quant_shards = Array[QuantViewType, num_devices](
                    uninitialized=True
                )
                var scale_shards = Array[ScaleViewType, num_devices](
                    uninitialized=True
                )
                var my_row_off = 0
                comptime for d in range(num_devices):
                    comptime d_group_start = ualign_down(d, group_size)
                    comptime assert conforms_to(
                        type_of(outputs_quant[d]), TileTensorable
                    )
                    comptime assert conforms_to(
                        type_of(outputs_scale[d]), TileTensorable
                    )
                    var q_base = rebind[QuantPtrType](
                        outputs_quant[d].to_tile_tensor[quant_dtype]().ptr
                    )
                    var s_base = rebind[ScalePtrType](
                        outputs_scale[d].to_tile_tensor[scales_dtype]().ptr
                    )
                    var row_off = 0
                    comptime for i in range(group_size):
                        var len_i = (
                            Int(in_tensors[d_group_start + i].num_elements())
                            // cols_rt
                        )
                        quant_views[d * group_size + i] = TileTensor(
                            q_base.unsafe_offset(row_off * cols_rt),
                            row_major(len_i, cols_rt),
                        )
                        scale_views[d * group_size + i] = TileTensor(
                            s_base.unsafe_offset(row_off * scale_cols),
                            row_major(len_i, scale_cols),
                        )
                        comptime if d_group_start + i == d:
                            quant_shards[d] = quant_views[d * group_size + i]
                            scale_shards[d] = scale_views[d * group_size + i]
                            comptime if d == index:
                                my_row_off = row_off
                        row_off += len_i

                var my_rows = Int(in_tensors[index].num_elements()) // cols_rt
                var my_in = TileTensor(
                    in_tensors[index].ptr, row_major(my_rows, cols_rt)
                )
                var my_sum = TileTensor(
                    sum_buf.ptr.unsafe_offset(my_row_off * cols_rt),
                    row_major(my_rows, cols_rt),
                )
                var my_normed = TileTensor(
                    normed_buf.ptr.unsafe_offset(my_row_off * cols_rt),
                    row_major(my_rows, cols_rt),
                )

                @inline(.always)
                def norm_input_fn[
                    width: Int
                ](coords: Coord) {var my_in, var my_sum} -> SIMD[dtype, width]:
                    var val = my_in.load[
                        width=width, alignment=align_of[dtype]()
                    ](coords)
                    my_sum.store[alignment=align_of[dtype]()](coords, val)
                    return val

                @inline(.always)
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) {
                    var my_normed
                } -> None:
                    my_normed.store[alignment=alignment](coords, val)

                rms_norm_gpu[
                    2,
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=True,
                ](
                    norm_input_fn,
                    norm_output_fn,
                    my_in.layout.shape_coord(),
                    gamma_tensor,
                    dev_ctxs_input[index],
                    dispatch_rows=rows_rt,
                )

                quantize_mx_amd(
                    dev_ctxs_input[index],
                    quant_shards[index],
                    scale_shards[index],
                    my_normed,
                )

                allgather[
                    dtype=quant_dtype, ngpus=num_devices, group_size=group_size
                ](
                    quant_shards,
                    quant_views,
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )
                allgather[
                    dtype=scales_dtype,
                    ngpus=num_devices,
                    group_size=group_size,
                ](
                    scale_shards,
                    scale_views,
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )

            @inline(.always)
            def two_launch_gated() raises {imm}:
                var min_group_rows = rows_rt
                comptime for g in range(num_devices // group_size):
                    var group_rows = 0
                    comptime for i in range(group_size):
                        group_rows += (
                            Int(in_tensors[g * group_size + i].num_elements())
                            // cols_rt
                        )
                    min_group_rows = min(min_group_rows, group_rows)
                if (
                    min_group_rows * cols_rt * size_of[dtype]()
                    >= AG_QUANT_ONLY_THRESHOLD
                ):
                    two_launch_quant_only()
                else:
                    two_launch_with_quant()

            comptime if gather_quant_only:
                _dispatch_ag_norm_quant[group_size=group_size](
                    in_tensors,
                    normed_buf,
                    sum_buf,
                    gamma_tensor,
                    epsilon,
                    weight_offset.cast[dtype](),
                    rank_sigs,
                    dev_ctxs_input[index],
                    mx_epilogue,
                    two_launch_gated,
                    my_rank=index,
                )
            else:
                _dispatch_ag_norm_quant[group_size=group_size](
                    in_tensors,
                    normed_buf,
                    sum_buf,
                    gamma_tensor,
                    epsilon,
                    weight_offset.cast[dtype](),
                    rank_sigs,
                    dev_ctxs_input[index],
                    mx_epilogue,
                    two_launch_with_quant,
                    my_rank=index,
                )

        _launch_device_collective[num_devices](
            launch_fused_ag_norm_quant, dev_ctxs_input.copy()
        )


@extensibility.register(
    "mo.composite.distributed.allgather_rms_norm_quant_mxfp6"
)
struct DistributedAllGatherRMSNormQuantMXFP6:
    """Registers `mo.composite.distributed.allgather_rms_norm_quant_mxfp6`."""

    @staticmethod
    def execute[
        Normed: TypeList[Trait=Output & Tensor, ...],
        Quant: TypeList[Trait=Output & Tensor, ...],
        Scales: TypeList[Trait=Output & Tensor, ...],
        Residuals: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        Gammas: TypeList[Trait=Input & Tensor, ...],
        //,
        _trace_name: StaticString,
        fp6_format: Int,
        epsilon: Float32,
        weight_offset: Float32,
        group_size: Int = 0,
    ](
        outputs_normed: TensorTuple[*Normed],
        outputs_quant: TensorTuple[*Quant],
        outputs_scale: TensorTuple[*Scales],
        outputs_residual: TensorTuple[*Residuals],
        inputs: TensorTuple[*Ins],
        signal_buffers: TensorTuple[*Sigs],
        gammas: TensorTuple[*Gammas],
        dev_ctxs_input: DeviceContextArray[_, _],
    ) raises:
        """Fused all-gather + RMSNorm that also emits a packed MXFP6 copy.

        `DistributedAllGatherRMSNorm` plus `outputs_quant`/`outputs_scale`. The
        quantize rides the collective's epilogue on the same bf16 that lands in
        `outputs_normed`, so it is byte-identical to a standalone quantize.

        Parameters:
            Normed: The element types of `outputs_normed` (inferred).
            Quant: The element types of `outputs_quant` (inferred).
            Scales: The element types of `outputs_scale` (inferred).
            Residuals: The element types of `outputs_residual` (inferred).
            Ins: The element types of `inputs` (inferred).
            Sigs: The element types of `signal_buffers` (inferred).
            Gammas: The element types of `gammas` (inferred).
            _trace_name: Trace name for profiling.
            fp6_format: FP6 encoding, 0 for E2M3 and 1 for E3M2.
            epsilon: RMSNorm epsilon, shared by every device.
            weight_offset: Gamma offset, shared by every device.
            group_size: Devices per independent all-gather group; see
                `DistributedAllGatherRMSNorm`.

        Args:
            outputs_normed: Per-device normed output (the group's gathered
                `[rows, cols]`, replicated within the group).
            outputs_quant: Per-device packed MXFP6 copy of `outputs_normed`,
                `[rows, cols * 3 // 4]`.
            outputs_scale: Per-device E8M0 block scales, `[rows, cols / 32]`.
            outputs_residual: Per-device gathered residual.
            inputs: Per-device input row-shards to gather.
            signal_buffers: Per-device synchronization buffers.
            gammas: Per-device RMSNorm gamma weights.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Targets AMD (CDNA4): the fallback calls `quantize_mx_amd` and
              the scale layout is its rank-2 one -- what
              `block_scaled_matmul_amd` takes as `a_scales`, not the SM100 SF
              atom and not the preshuffled order the `_preb` kernel needs.
            - Maximum of 8 GPUs; requires P2P.
        """
        comptime num_devices = len(Ins)
        comptime dtype = Ins[0].dtype
        comptime quant_dtype = Quant[0].dtype
        comptime scales_dtype = Scales[0].dtype
        comptime rank = Ins[0].rank
        comptime assert len(Sigs) == num_devices, (
            "expected allgather_rms_norm_quant_mxfp8 inputs and signal buffers"
            " to have the same number of elements"
        )
        _check_tensor_tuple[num_devices, dtype, rank, *Ins]()
        _check_tensor_tuple[num_devices, dtype, rank, *Normed]()
        _check_tensor_tuple[num_devices, quant_dtype, rank, *Quant]()
        _check_tensor_tuple[num_devices, scales_dtype, rank, *Scales]()
        _check_tensor_tuple[num_devices, dtype, rank, *Residuals]()
        _check_tensor_tuple[num_devices, .uint8, 1, *Sigs]()
        _check_tensor_tuple[num_devices, dtype, 1, *Gammas]()
        comptime assert group_size >= 2, "group_size must be at least 2"
        comptime assert (
            num_devices % group_size == 0
        ), "group_size must evenly divide the number of devices"
        comptime assert quant_dtype == .uint8, (
            "MXFP6 quant output must be uint8 (four packed codes per three"
            " bytes)"
        )
        comptime assert (
            scales_dtype == .float8_e8m0fnu
        ), "MXFP6 block scales must be float8_e8m0fnu"

        _check_signal_buffer_size(Int(signal_buffers[0].shape().product()), 0)

        comptime assert fp6_format in (
            0,
            1,
        ), "fp6_format must be 0 (E2M3) or 1 (E3M2)"
        comptime fp6_fmt = FP6Format.E2M3 if fp6_format == 0 else FP6Format.E3M2

        # World-view input tensors and signals, indexed by GLOBAL device
        # rank; see `DistributedAllGatherRMSNorm`.
        comptime InputTensorType = TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, rank].element_types,
                stride_types=DynamicCoord[.int64, rank].element_types,
            ],
            ImmUntrackedOrigin,
        ]
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
                inputs[i].to_tile_tensor[dtype]().make_dynamic[.int64]()
            )
        _check_uniform_cols[op="allgather_rms_norm_quant_mxfp6"](inputs)

        # `allgather`'s world-view output array holds every device's own
        # `group_size` output windows. Every slot has to name a real window,
        # not just this device's: a grouped allgather may route part of a
        # shard through the paired group, whose GPUs then write these buffers
        # directly. Each device windows its own output by ITS group's ragged
        # row counts. Built once here; nothing in it depends on which device
        # is launching, and the column count is uniform across the world.
        comptime assert conforms_to(
            type_of(outputs_residual[0]), TileTensorable
        )
        var world_cols = Int(
            outputs_residual[0].to_tile_tensor[dtype]().dim[rank - 1]()
        )
        comptime OutViewType = type_of(
            TileTensor(
                rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                    outputs_residual[0].to_tile_tensor[dtype]().ptr
                ),
                row_major(world_cols, world_cols),
            )
        )
        var world_out_views = Array[OutViewType, num_devices * group_size](
            uninitialized=True
        )
        comptime for d in range(num_devices):
            comptime d_group_start = ualign_down(d, group_size)
            comptime assert conforms_to(
                type_of(outputs_residual[d]), TileTensorable
            )
            var d_base = rebind[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
                outputs_residual[d].to_tile_tensor[dtype]().ptr
            )
            var row_off = 0
            comptime for i in range(group_size):
                var len_i = (
                    Int(in_tensors[d_group_start + i].num_elements())
                    // world_cols
                )
                world_out_views[d * group_size + i] = TileTensor(
                    d_base.unsafe_offset(row_off * world_cols),
                    row_major(len_i, world_cols),
                )
                row_off += len_i

        @inline(.always)
        def launch_fused_ag_norm_quant[
            index: Int
        ]() raises {
            imm in_tensors,
            imm rank_sigs,
            imm world_out_views,
            imm dev_ctxs_input,
            imm gammas,
            imm outputs_normed,
            imm outputs_quant,
            imm outputs_scale,
            imm outputs_residual,
        }:
            # `local_rank`/`group_start`: see `DistributedAllGatherRMSNorm`.
            comptime group_id, local_rank = divmod(index, group_size)
            comptime group_start = group_id * group_size

            comptime assert conforms_to(
                type_of(outputs_normed[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_quant[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_scale[index]), TileTensorable
            )
            comptime assert conforms_to(
                type_of(outputs_residual[index]), TileTensorable
            )
            comptime assert conforms_to(type_of(gammas[index]), TileTensorable)
            var normed_buf = outputs_normed[index].to_tile_tensor[dtype]()
            var quant_buf = outputs_quant[index].to_tile_tensor[quant_dtype]()
            var scale_buf = outputs_scale[index].to_tile_tensor[scales_dtype]()
            var sum_buf = outputs_residual[index].to_tile_tensor[dtype]()
            var gamma_tensor = gammas[index].to_tile_tensor[dtype]()
            var cols_rt = Int(sum_buf.dim[rank - 1]())

            var rows_rt = normed_buf.num_elements() // cols_rt
            var quant_cols = Int(quant_buf.dim[rank - 1]())
            var packed_cols = (cols_rt * 3) // 4
            if (
                cols_rt % 4 != 0
                or quant_cols != packed_cols
                or quant_buf.num_elements() != rows_rt * packed_cols
            ):
                raise Error(
                    String(
                        "allgather_rms_norm_quant_mxfp6: outputs_quant is ",
                        quant_buf.num_elements() // quant_cols,
                        " x ",
                        quant_cols,
                        ", expected ",
                        rows_rt,
                        " x ",
                        packed_cols,
                        " (normed cols * 3 / 4)",
                    )
                )
            var scale_cols = Int(scale_buf.dim[rank - 1]())
            if (
                scale_cols * MXFP6_SF_VECTOR_SIZE != cols_rt
                or scale_buf.num_elements() != rows_rt * scale_cols
            ):
                raise Error(
                    String(
                        "allgather_rms_norm_quant_mxfp6: outputs_scale is ",
                        scale_buf.num_elements() // scale_cols,
                        " x ",
                        scale_cols,
                        ", expected ",
                        rows_rt,
                        " x ",
                        cols_rt // MXFP6_SF_VECTOR_SIZE,
                        " (normed cols / ",
                        MXFP6_SF_VECTOR_SIZE,
                        ")",
                    )
                )

            # The `{var}` copies are REQUIRED: a local this closure captures by
            # reference reaches the device as a host-stack pointer and faults.
            @inline(.always)
            def mx_epilogue[
                width: Int
            ](row: Int, col: Int, val: SIMD[dtype, width]) {
                var quant_buf, var scale_buf, var cols_rt
            }:
                comptime assert width % 4 == 0, (
                    "packed FP6 addresses groups of four codes, so a thread's"
                    " slice must be a whole number of groups"
                )
                var codes: SIMD[.uint8, width]
                var e8m0: Scalar[scales_dtype]
                codes, e8m0 = quantize_mxfp6_lane_group[
                    scales_dtype,
                    fp6_fmt,
                    SF_VECTOR_SIZE=MXFP6_SF_VECTOR_SIZE,
                ](val)
                if col < cols_rt:
                    var byte_col = (col * 3) // 4
                    comptime for g in range(width // 4):
                        var word = pack_fp6_x4(codes.slice[4, offset=g * 4]())
                        comptime for b in range(3):
                            quant_buf.store(
                                (row, byte_col + g * 3 + b),
                                rebind[Scalar[quant_dtype]](
                                    UInt8(
                                        (word >> UInt32(8 * b)) & UInt32(0xFF)
                                    )
                                ),
                            )
                    if col % MXFP6_SF_VECTOR_SIZE == 0:
                        scale_buf[row, col // MXFP6_SF_VECTOR_SIZE] = e8m0

            @inline(.always)
            def two_launch_with_quant() raises {imm}:
                allgather[
                    dtype=dtype, ngpus=num_devices, group_size=group_size
                ](
                    in_tensors,
                    world_out_views,
                    rank_sigs,
                    dev_ctxs_input[index],
                    index,
                )

                @inline(.always)
                def norm_input_fn[
                    width: Int
                ](coords: Coord) {var sum_buf} -> SIMD[dtype, width]:
                    return sum_buf.load[
                        width=width, alignment=align_of[dtype]()
                    ](coords)

                @inline(.always)
                def norm_output_fn[
                    width: SIMDLength, alignment: Int
                ](coords: Coord, val: SIMD[dtype, width]) {
                    var normed_buf
                } -> None:
                    normed_buf.store[alignment=alignment](coords, val)

                rms_norm_gpu[
                    rank,
                    epsilon=epsilon,
                    weight_offset=weight_offset,
                    multiply_before_cast=True,
                ](
                    norm_input_fn,
                    norm_output_fn,
                    sum_buf.layout.shape_coord(),
                    gamma_tensor,
                    dev_ctxs_input[index],
                )

                quantize_mxfp6_amd[fp6_fmt](
                    dev_ctxs_input[index],
                    quant_buf,
                    scale_buf,
                    normed_buf.as_imm(),
                )

            _dispatch_ag_norm_quant[group_size=group_size](
                in_tensors,
                normed_buf,
                sum_buf,
                gamma_tensor,
                epsilon,
                weight_offset.cast[dtype](),
                rank_sigs,
                dev_ctxs_input[index],
                mx_epilogue,
                two_launch_with_quant,
                my_rank=index,
            )

        _launch_device_collective[num_devices](
            launch_fused_ag_norm_quant, dev_ctxs_input.copy()
        )


@extensibility.register("lamport_allreduce_rmsnorm")
struct LamportAllreduceRMSNorm:
    """Per-rank fused Lamport allreduce + RMSNorm (high-perf protocol).

    Built on `comm.allreduce_lamport_rmsnorm`; the Lamport comm region is
    embedded in `Signal` (`Signal.lamport_region`), so the caller sizes/
    initializes the signal buffers as `sizeof(Signal)` bytes. `ngpus` is
    inferred from the number of signal buffers passed in (must be in [2, 8]).

    Shape constraints (also checked inside `lamport_allreduce_rmsnorm`, but
    surfaced here for op-level diagnostics):
    - `cols % atomic_width == 0`  (whole 128-bit Lamport packs only;
      `atomic_width = 16 / size_of[dtype]`, so bf16 needs `cols % 8 == 0`).
    - `cols / atomic_width <= BLOCK_SIZE`  (one pack per thread / row;
      `BLOCK_SIZE = floor(max_tpb / WARP_SIZE) * WARP_SIZE`, e.g. 1024 on
      Hopper/Blackwell, capping bf16 hidden at 8192).
    """

    @staticmethod
    def execute[
        Sigs: TypeList[Trait=MutableInput & Tensor, ...],
        //,
        my_rank: Int,
    ](
        output: Some[Output & Tensor],
        act: Some[Input & Tensor],
        gamma: Some[Input & Tensor],
        signal_buffers: TensorTuple[*Sigs],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            ctx.target.is_gpu()
        ), "lamport_allreduce_rmsnorm: gpu only"
        comptime ngpus = len(Sigs)
        comptime assert (
            ngpus >= 2 and ngpus <= MAX_GPUS
        ), "lamport_allreduce_rmsnorm: signal_buffers.size must be in [2, 8]"
        comptime assert (
            my_rank >= 0 and my_rank < ngpus
        ), "lamport_allreduce_rmsnorm: my_rank must be in [0, ngpus)"
        comptime epsilon = Float32(1e-6)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(act).dtype == dtype
        comptime assert type_of(act).rank == 2
        comptime assert type_of(gamma).dtype == dtype
        comptime assert type_of(gamma).rank == 1
        _check_tensor_tuple[ngpus, .uint8, 1, *Sigs]()
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(act), TileTensorable)
        comptime assert conforms_to(type_of(gamma), TileTensorable)

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

        var rank_sigs = Array[_, ngpus](fill_with_unrolled=rank_sig)

        var rows = Int(act.shape()[0].value())
        var cols = Int(act.shape()[1].value())

        # Surface kernel shape constraints at the op boundary so the failure is
        # attributed to the op rather than to the kernel-host launcher. Mirrors
        # the checks in `comm.allreduce_lamport_rmsnorm`.
        comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
        comptime max_tpb = ctx.default_device_info.max_thread_block_size
        comptime BLOCK_SIZE = align_down(max_tpb, WARP_SIZE)
        if cols % atomic_width != 0:
            raise Error(
                "lamport_allreduce_rmsnorm: cols (",
                cols,
                ") must be a multiple of atomic_width (",
                atomic_width,
                ") -- whole 128-bit Lamport packs required",
            )
        if cols // atomic_width > BLOCK_SIZE:
            raise Error(
                "lamport_allreduce_rmsnorm: cols/atomic_width (",
                cols // atomic_width,
                ") exceeds BLOCK_SIZE (",
                BLOCK_SIZE,
                ") -- one pack per thread required",
            )

        var src = rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            act.to_tile_tensor[dtype]().ptr
        )
        var dst = rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
            output.to_tile_tensor[dtype]().ptr
        )
        var gm = rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            gamma.to_tile_tensor[dtype]().ptr
        )

        lamport_allreduce_rmsnorm[dtype, ngpus](
            my_rank,
            src,
            dst,
            gm,
            rank_sigs,
            rows,
            cols,
            epsilon.cast[dtype](),
            ctx,
        )
