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

"""Grouped collectives compile when device groups carry different shapes.

Each handler in ``distributed.mojo`` packs a world-view array of per-device
tensors, whose element type it derives from device 0. Baking device 0's static
extents into that type makes the ``rebind`` for a group with a different row
count fail KGEN elaboration, which is a compile-time failure the execution
tests in ``max/tests/integration/graph/multi_gpu`` can only report on a machine
with four GPUs. Virtual devices reproduce it with none.

Virtual-device mode latches process-wide at the first device creation, so the
knobs below are set at import and this needs a target of its own.
"""

from __future__ import annotations

from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

NUM_GPUS = 4
GROUP_SIZE = 2

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(NUM_GPUS)

import pytest
from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType
from max.driver import CPU, Accelerator, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.type import _ChainType
from max.graph.value import _ChainValue
from max.nn import Signals

N = 1024  # Allreduce/reduce-scatter column count.
COLS = 6144  # Hidden size the fused RMSNorm composites are tuned for.
EPS = 1e-6
WEIGHT_OFFSET = 1.0


def _signals() -> Signals:
    return Signals(devices=[DeviceRef.GPU(id=i) for i in range(NUM_GPUS)])


def _compile(graph: Graph) -> None:
    """Loads ``graph`` for four virtual accelerators.

    Args:
        graph: The graph to compile.
    """
    assert is_virtual_device_mode()
    session = InferenceSession(
        devices=[CPU()] + [Accelerator(i) for i in range(NUM_GPUS)]
    )
    session.load(graph)


def _allreduce_graph(rows_per_device: list[int], group_size: int) -> Graph:
    """Builds an allreduce graph with a per-device row count.

    Args:
        rows_per_device: Rows the tensor on each device carries.
        group_size: Devices per independent allreduce group.

    Returns:
        The graph.
    """
    signals = _signals()
    input_types = [
        TensorType(
            dtype=DType.float32,
            shape=[rows_per_device[i], N],
            device=signals.devices[i],
        )
        for i in range(NUM_GPUS)
    ]
    with Graph(
        "grouped_allreduce",
        input_types=[*input_types, *signals.input_types()],
    ) as graph:
        outputs = ops.allreduce.sum(
            [graph.inputs[i].tensor for i in range(NUM_GPUS)],
            [inp.buffer for inp in graph.inputs[NUM_GPUS:]],
            group_size=group_size,
        )
        graph.output(*outputs)
        return graph


def _reducescatter_axis0_graph(rows_per_device: list[int]) -> Graph:
    """Builds a grouped reduce-scatter graph that scatters rows.

    Args:
        rows_per_device: Rows the tensor on each device carries.

    Returns:
        The graph.
    """
    signals = _signals()
    input_types = [
        TensorType(
            dtype=DType.float32,
            shape=[rows_per_device[i], N],
            device=signals.devices[i],
        )
        for i in range(NUM_GPUS)
    ]
    with Graph(
        "grouped_reducescatter_axis0",
        input_types=[*input_types, *signals.input_types()],
    ) as graph:
        outputs = ops.reducescatter.sum(
            [graph.inputs[i].tensor for i in range(NUM_GPUS)],
            [inp.buffer for inp in graph.inputs[NUM_GPUS:]],
            axis=0,
            group_size=GROUP_SIZE,
        )
        graph.output(*outputs)
        return graph


def _allgather_rms_norm_graph(rows_per_device: list[int]) -> Graph:
    """Builds a grouped fused all-gather + RMSNorm graph.

    Args:
        rows_per_device: Rows the shard on each device carries.

    Returns:
        The graph.
    """
    signals = _signals()
    shard_types = [
        TensorType(
            dtype=DType.bfloat16,
            shape=[rows_per_device[i], COLS],
            device=device,
        )
        for i, device in enumerate(signals.devices)
    ]
    gamma_types = [
        TensorType(dtype=DType.bfloat16, shape=[COLS], device=device)
        for device in signals.devices
    ]
    with Graph(
        "grouped_allgather_rms_norm",
        input_types=[*shard_types, *gamma_types, *signals.input_types()],
    ) as graph:
        normed, residual = ops.allgather_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:NUM_GPUS]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * NUM_GPUS :]],
            gammas=[v.tensor for v in graph.inputs[NUM_GPUS : 2 * NUM_GPUS]],
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
            group_size=GROUP_SIZE,
        )
        graph.output(*normed, *residual)
        return graph


def _reduce_scatter_rms_norm_graph(rows_per_device: list[int]) -> Graph:
    """Builds a grouped fused reduce-scatter + RMSNorm graph with a residual.

    Args:
        rows_per_device: Rows the activation on each device carries.

    Returns:
        The graph.
    """
    signals = _signals()
    act_types = [
        TensorType(
            dtype=DType.bfloat16,
            shape=[rows_per_device[i], COLS],
            device=device,
        )
        for i, device in enumerate(signals.devices)
    ]
    gamma_types = [
        TensorType(dtype=DType.bfloat16, shape=[COLS], device=device)
        for device in signals.devices
    ]
    with Graph(
        "grouped_reduce_scatter_rms_norm",
        input_types=[
            *act_types,
            *gamma_types,
            *act_types,
            *signals.input_types(),
        ],
    ) as graph:
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:NUM_GPUS]],
            signal_buffers=[v.buffer for v in graph.inputs[3 * NUM_GPUS :]],
            gammas=[v.tensor for v in graph.inputs[NUM_GPUS : 2 * NUM_GPUS]],
            epsilon=EPS,
            residuals=[
                v.tensor for v in graph.inputs[2 * NUM_GPUS : 3 * NUM_GPUS]
            ],
            weight_offset=WEIGHT_OFFSET,
            group_size=GROUP_SIZE,
        )
        graph.output(*normed, *residual)
        return graph


def _allgather_rms_norm_graph_unvalidated(cols_per_device: list[int]) -> Graph:
    """Stages the fused all-gather + RMSNorm op without the builder's checks.

    Args:
        cols_per_device: Columns the shard on each device carries.

    Returns:
        The graph.
    """
    signals = _signals()
    shard_types = [
        TensorType(
            dtype=DType.bfloat16,
            shape=[4, cols_per_device[i]],
            device=device,
        )
        for i, device in enumerate(signals.devices)
    ]
    gamma_types = [
        TensorType(
            dtype=DType.bfloat16, shape=[cols_per_device[i]], device=device
        )
        for i, device in enumerate(signals.devices)
    ]
    with Graph(
        "grouped_allgather_rms_norm_unvalidated",
        input_types=[*shard_types, *gamma_types, *signals.input_types()],
    ) as graph:
        gathered_types = [
            TensorType(
                dtype=DType.bfloat16,
                shape=[4 * GROUP_SIZE, cols_per_device[i]],
                device=device,
            )
            for i, device in enumerate(signals.devices)
        ]
        eps = ops.constant(EPS, DType.float32, DeviceRef.CPU())
        offset = ops.constant(WEIGHT_OFFSET, DType.bfloat16, DeviceRef.CPU())
        *results, out_chain = graph._add_op_generated(
            mo.CompositeDistributedAllgatherRmsNormOp,
            gathered_types,
            gathered_types,
            _ChainType(),
            [v.tensor for v in graph.inputs[:NUM_GPUS]],
            [v.buffer for v in graph.inputs[2 * NUM_GPUS :]],
            [v.tensor for v in graph.inputs[NUM_GPUS : 2 * NUM_GPUS]],
            [eps] * NUM_GPUS,
            [offset] * NUM_GPUS,
            graph.device_chains.merge_for(signals.devices),
            IntegerAttr(IntegerType(64), GROUP_SIZE),
        )
        assert isinstance(out_chain, _ChainValue)
        graph._update_chain(out_chain)
        graph.output(*(r.tensor for r in results))
        return graph


def test_ungrouped_allreduce_compiles() -> None:
    """Control: one world-wide group, so every device shares a shape."""
    _compile(_allreduce_graph([5] * NUM_GPUS, group_size=NUM_GPUS))


def test_grouped_allreduce_equal_shapes_compiles() -> None:
    """Control: grouping alone, with the groups' shapes still matching."""
    _compile(_allreduce_graph([5] * NUM_GPUS, group_size=GROUP_SIZE))


def test_grouped_allreduce_mixed_shapes_compiles() -> None:
    _compile(_allreduce_graph([5, 5, 9, 9], group_size=GROUP_SIZE))


def test_grouped_reducescatter_axis0_mixed_shapes_compiles() -> None:
    _compile(_reducescatter_axis0_graph([5, 5, 3, 3]))


def test_grouped_allgather_rms_norm_mixed_shapes_compiles() -> None:
    # The groups sit on opposite sides of the fuse threshold, as in the
    # execution test, so both dispatch branches are elaborated.
    _compile(_allgather_rms_norm_graph([4, 4, 128, 128]))


def test_grouped_reduce_scatter_rms_norm_ragged_compiles() -> None:
    # 5 rows shard to 3,2 and 3 rows to 2,1: the reduced-sum outputs differ
    # both across groups and within one.
    _compile(_reduce_scatter_rms_norm_graph([5, 5, 3, 3]))


def test_cross_group_column_mismatch_rejected_by_handler() -> None:
    """The handler, not just the builder, refuses groups with other columns.

    The graph is staged below the builder's validation, so this reaches the
    static column check in ``distributed.mojo``.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _compile(
            _allgather_rms_norm_graph_unvalidated(
                [COLS, COLS, 2 * COLS, 2 * COLS]
            )
        )
    # The engine wraps the compile failure; the kernel's message is the cause.
    assert "same column count on every device" in str(excinfo.value.__cause__)
