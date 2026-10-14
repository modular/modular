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

"""Grouped quantizing all-gather + RMSNorm compiles with per-group row counts.

The MXFP8 and MXFP6 handlers pack the same device-0-typed world-view array as
the plain all-gather norm, so they carried the same elaboration failure. Their
quantize step rides the CDNA4 path, hence the virtual AMD target. Virtual-device
mode latches process-wide at the first device creation, so the knobs below are
set at import and this needs a target of its own.
"""

from __future__ import annotations

from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

NUM_GPUS = 4
GROUP_SIZE = 2

set_virtual_device_api("hip")
set_virtual_device_target_arch("gfx950")
set_virtual_device_count(NUM_GPUS)

from collections.abc import Callable

import pytest
from max.driver import CPU, Accelerator, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Signals

COLS = 6144  # Hidden size the fused composites are tuned for.
EPS = 1e-6
WEIGHT_OFFSET = 1.0

QuantOp = Callable[..., tuple[list[TensorValue], ...]]


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


def _quant_graph(op: QuantOp, rows_per_device: list[int]) -> Graph:
    """Builds a grouped quantizing all-gather + RMSNorm graph.

    Args:
        op: The quantizing op to stage.
        rows_per_device: Rows the shard on each device carries.

    Returns:
        The graph.
    """
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(NUM_GPUS)])
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
        "grouped_allgather_rms_norm_quant",
        input_types=[*shard_types, *gamma_types, *signals.input_types()],
    ) as graph:
        normed, quant, scales, residual = op(
            inputs=[v.tensor for v in graph.inputs[:NUM_GPUS]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * NUM_GPUS :]],
            gammas=[v.tensor for v in graph.inputs[NUM_GPUS : 2 * NUM_GPUS]],
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
            group_size=GROUP_SIZE,
        )
        graph.output(*normed, *residual, *quant, *scales)
        return graph


QUANT_OPS = [
    pytest.param(ops.allgather_rms_norm_quant_mxfp8, id="mxfp8"),
    pytest.param(ops.allgather_rms_norm_quant_mxfp6, id="mxfp6"),
]


@pytest.mark.parametrize("op", QUANT_OPS)
def test_grouped_quant_equal_shapes_compiles(op: QuantOp) -> None:
    """Control: grouping alone, with the groups' shapes still matching."""
    _compile(_quant_graph(op, [128] * NUM_GPUS))


@pytest.mark.parametrize("op", QUANT_OPS)
def test_grouped_quant_mixed_shapes_compiles(op: QuantOp) -> None:
    # The groups sit on opposite sides of the fuse threshold, so both
    # dispatch branches are elaborated.
    _compile(_quant_graph(op, [4, 4, 128, 128]))
