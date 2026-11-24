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
"""Op implementation for all_to_all."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..dim import StaticDim
from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def all_to_all(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    split_axis: int = 0,
    concat_axis: int = 0,
    group_size: int | None = None,
) -> list[TensorValue]:
    """Collective all-to-all: each device sends one chunk to every peer.

    Each device splits its input into ``group_size`` equal chunks along
    ``split_axis`` and sends chunk ``j`` to rank ``j`` of its group. Each
    device's output concatenates the chunks it receives, in source rank
    order, along ``concat_axis``. With both axes ``0`` this is
    ``torch.distributed.all_to_all_single``; with different axes it moves a
    tensor sharded along ``concat_axis`` to one sharded along
    ``split_axis``. It is one launch: every device reads its chunks from its
    peers' inputs directly.

    Args:
        inputs: The input tensors, one per device. Shapes must match within
            a group, and ``split_axis`` must divide by the group size.
        signal_buffers: Device buffer values used for synchronization.
        split_axis: The input axis split into one chunk per peer.
        concat_axis: The output axis the received chunks concatenate along.
        group_size: The number of contiguous devices per independent group.
            Defaults to all devices.

    Returns:
        One output per device: its input's shape with ``split_axis`` divided
        by the group size and ``concat_axis`` multiplied by it.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    num_devices = len(inputs)
    if num_devices == 0:
        raise ValueError("all_to_all requires at least one input")
    if len(signal_buffers) != num_devices:
        raise ValueError(
            f"expected number of inputs ({num_devices}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )
    group_size = group_size or num_devices
    if group_size < 2 or num_devices % group_size != 0:
        raise ValueError(
            "all_to_all requires a group_size of at least 2 that evenly "
            f"divides the number of inputs. Got: {group_size=} and "
            f"{num_devices=}"
        )
    devices = [input.device for input in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "all_to_all must have unique devices across its input tensors. "
            f"Got: {devices=}"
        )
    dtype = inputs[0].dtype
    rank = inputs[0].shape.rank
    if any(
        input.dtype != dtype or input.shape.rank != rank for input in inputs
    ):
        raise ValueError(
            "all_to_all must have the same dtype and rank across all input "
            f"tensors. Got: {inputs=}"
        )
    split_axis = split_axis + rank if split_axis < 0 else split_axis
    concat_axis = concat_axis + rank if concat_axis < 0 else concat_axis
    if not (0 <= split_axis < rank and 0 <= concat_axis < rank):
        raise ValueError(
            f"all_to_all axes ({split_axis}, {concat_axis}) are out of bounds "
            f"for rank {rank}"
        )
    for group_start in range(0, num_devices, group_size):
        group = inputs[group_start : group_start + group_size]
        if any(input.shape != group[0].shape for input in group[1:]):
            raise ValueError(
                "all_to_all must have the same shape across the inputs of "
                f"each group. Got: {inputs=}"
            )

    output_types = []
    for input in inputs:
        shape = list(input.shape)
        split = shape[split_axis]
        if isinstance(split, StaticDim) and int(split) % group_size != 0:
            raise ValueError(
                f"all_to_all cannot split axis {split_axis} of size "
                f"{int(split)} into {group_size} chunks"
            )
        shape[split_axis] = split // group_size
        shape[concat_axis] = shape[concat_axis] * group_size
        output_types.append(
            TensorType(dtype=dtype, shape=shape, device=input.device)
        )

    graph = Graph.current
    in_chain = graph.device_chains.merge_for(devices)
    i64 = IntegerType(64)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedAllToAllOp,
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        in_chain,
        IntegerAttr(i64, split_axis),
        IntegerAttr(i64, concat_axis),
        IntegerAttr(i64, group_size),
    )
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain
    return [res.tensor for res in results]
