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
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License, v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""KDA prefill routing in Kimi K3's ModuleV3 layer, read off the graph.

The layer builds one ``kda_decode`` call for every batch and carries no
device knowledge: no accelerator query, no servability check, no per-batch
conditional, no operand casts. On gfx950 the op itself runs the
chunk-parallel pipeline for the prefill batches its contract covers and the
token-sequential kernel for everything else; that dispatch is pinned on
gfx950 in ``test_kda_chunk_prefill.py``, and this target checks only what
the layer puts in the graph.

The recurrence's graph function is only built here, never compiled, and is
accelerator-independent: the virtual gfx950 device exists only so traced
inputs can resolve a device on a CPU-only host. The tensor-parallel layer
itself is traced on virtual devices in its own target,
``test_kda_prefill_routing_layer.py``, because a process that has already
opened a device cannot switch to virtual ones.

Whatever the gate arm or the pool dtype, the graph must hold exactly one
``kda_decode`` and nothing else recurrent: no ``kda_chunk``, no ``mo.if``
-- the op reads those at execute time, so the graph cannot branch on them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.legacy.state_space.kimi_delta import KdaGateMode
from max.pipelines.architectures.kimi_k3_modulev3.layers import (
    kda_attention,
)
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context

HEADS = 12
"""One TP8 rank's KDA heads: K3's 96 heads split eight ways."""

HEAD_DIM = 128
BATCH = 3
SLOTS = 6
GPU = DeviceRef.GPU()


@contextmanager
def _virtual_gpu() -> Iterator[None]:
    """One virtual gfx950 device, so tracing needs no GPU on the host.

    Wrapping graph inputs as traced tensors resolves their device to a
    driver ``Accelerator``; without a virtual one that fails on CPU-only CI
    hosts.
    """
    set_virtual_device_api("hip")
    set_virtual_device_target_arch("gfx950")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def _recurrence_mlir(
    gate_mode: KdaGateMode = "safe",
    pool_dtype: DType = DType.float32,
) -> str:
    """Builds the recurrence's graph and returns its MLIR.

    The token axis stays symbolic, as the model graph's does: the op reads
    it at execute time, when it picks the recurrence, not at build time.
    """
    tokens = "total_tokens"
    io_type = TensorType(DType.bfloat16, [1, tokens, HEADS, HEAD_DIM], GPU)
    with (
        Graph(
            "k3_kda_recurrence",
            input_types=[
                io_type,
                io_type,
                io_type,
                TensorType(DType.float32, [1, tokens, HEADS, HEAD_DIM], GPU),
                TensorType(DType.float32, [1, tokens, HEADS], GPU),
                TensorType(DType.float32, [HEADS], GPU),
                TensorType(DType.float32, [HEADS, HEAD_DIM], GPU),
                TensorType(DType.uint32, [BATCH + 1], GPU),
                BufferType(pool_dtype, [SLOTS, HEADS, HEAD_DIM, HEAD_DIM], GPU),
                TensorType(DType.uint32, [BATCH], GPU),
            ],
        ) as graph,
        ensure_context(),
        _virtual_gpu(),
    ):
        # The layer's functional op takes traced tensors, as it does in the
        # model's graph, so the graph inputs are wrapped before the call.
        tensors = [
            *(Tensor.from_graph_value(i.tensor) for i in graph.inputs[:8]),
            Tensor.from_graph_value(graph.inputs[8].buffer),
            Tensor.from_graph_value(graph.inputs[9].tensor),
        ]
        out = kda_attention.kda_decode(
            *tensors,
            gate_mode=gate_mode,
            out_type=io_type,
        )
        graph.output(TensorValue(out))
    return str(graph)


def _chunk_lines(mlir: str) -> list[str]:
    return [line for line in mlir.splitlines() if '"kda_chunk"' in line]


def _decode_lines(mlir: str) -> list[str]:
    return [line for line in mlir.splitlines() if '"kda_decode"' in line]


def _assert_one_decode(mlir: str) -> None:
    assert len(_decode_lines(mlir)) == 1, (
        "the recurrence must be exactly one kda_decode call"
    )
    assert not _chunk_lines(mlir), (
        "the layer must not route to kda_chunk: the kda_decode op picks"
        " the recurrence itself"
    )
    assert "mo.if" not in mlir, "the layer needs no per-batch conditional"


def test_recurrence_is_one_kda_decode() -> None:
    """One ``kda_decode`` in the graph, nothing else recurrent."""
    _assert_one_decode(_recurrence_mlir())


@pytest.mark.parametrize(
    ("gate_mode", "pool_dtype"),
    [
        ("original", DType.float32),
        ("safe", DType.bfloat16),
        ("original", DType.bfloat16),
    ],
)
def test_gate_and_pool_dtype_change_nothing(
    gate_mode: KdaGateMode, pool_dtype: DType
) -> None:
    """The graph is the same whatever the gate arm or the pool dtype.

    These are the shapes the AMD chunk contract excludes -- the "original"
    gate, a bf16 pool -- and a layer that routed on them would grow a
    conditional or a second op; the op reads them at execute time instead,
    so the graph cannot.
    """
    _assert_one_decode(_recurrence_mlir(gate_mode, pool_dtype))
