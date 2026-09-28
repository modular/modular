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

"""Gated DeltaNet recurrence routing, read off the built graph.

The kernel-level differential test grades ``kda_chunk_launch`` on tensors it
builds itself, which says nothing about what this layer hands the op. What can
go wrong here is wiring rather than arithmetic, and it goes wrong quietly: a
gate spread that lowers to a view still produces the right *values* -- every
channel of a head holds the same scalar -- so a shape-only or output-only check
passes while the descriptor reads past its buffer.
"""

from __future__ import annotations

import pytest
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn import Module
from max.pipelines.architectures.qwen3_5.layers.gated_deltanet import (
    GatedDeltaNet,
)

DEVICE = DeviceRef.GPU(0)
HIDDEN = 5120
NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
HEAD_DIM = 128
CONV_KERNEL = 4
SLOTS = 4


def _qualify_weight_names(layer: Module, prefix: str = "gdn") -> None:
    """Give every weight its module path, the way a parent module would.

    Built standalone, the layer holds several weights named plainly "weight"
    (the norm's and the projections'), and the graph refuses a duplicate. In a
    real model the enclosing modules supply the path.
    """
    for name, weight in layer.layer_weights.items():
        weight.name = f"{prefix}.{name}"
    for name, sub in layer.sublayers.items():
        _qualify_weight_names(sub, f"{prefix}.{name}")


def _forward_mlir() -> str:
    layer = GatedDeltaNet(
        hidden_size=HIDDEN,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        dtype=DType.bfloat16,
        device=DEVICE,
    )
    _qualify_weight_names(layer)
    conv_dim = NUM_KEY_HEADS * HEAD_DIM * 2 + NUM_VALUE_HEADS * HEAD_DIM
    with Graph(
        "gdn_routing",
        input_types=[
            TensorType(DType.bfloat16, ["total_tokens", HIDDEN], device=DEVICE),
            BufferType(
                DType.float32, [SLOTS, conv_dim, CONV_KERNEL - 1], device=DEVICE
            ),
            TensorType(DType.uint32, ["batch"], device=DEVICE),
            BufferType(
                DType.float32,
                [SLOTS, NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM],
                device=DEVICE,
            ),
            TensorType(DType.uint32, ["batch"], device=DEVICE),
            TensorType(DType.uint32, ["batch_plus_one"], device=DEVICE),
        ],
    ) as graph:
        x, conv_pool, conv_row, rec_pool, rec_row, offsets = graph.inputs
        out = layer(
            x.tensor,
            conv_pool.buffer,
            conv_row.tensor,
            rec_pool.buffer,
            rec_row.tensor,
            offsets.tensor,
        )
        graph.output(out)
    return str(graph)


# The layer only names the chunk op on an NVIDIA accelerator, so the
# assertions below describe CUDA builds alone.
cuda_only = pytest.mark.skipif(
    accelerator_api() != "cuda",
    reason="gated-DeltaNet prefill takes the chunk op on NVIDIA only",
)


@cuda_only
def test_prefill_and_decode_are_both_wired() -> None:
    """Both recurrences are present, behind a conditional."""
    mlir = _forward_mlir()
    assert "kda_chunk" in mlir, "prefill is not routed to the fused chunk op"
    assert "gated_delta_recurrence" in mlir, (
        "decode must stay on the sequential recurrence"
    )
    assert "mo.if" in mlir, (
        "the two recurrences must sit behind a conditional, not run in series"
    )


@cuda_only
def test_gate_spread_is_materialized_not_a_view() -> None:
    """The gate spread must not lower to a broadcast view.

    ``mo.broadcast_to`` carries ``MO_ViewLike``, so its result may keep a
    stride-0 buffer holding one column per head. The chunk op reads the gate
    through a TMA descriptor over a flat ``[token, HV*K]`` shape and would walk
    ``K`` times past the end of such a buffer. An elementwise multiply is not
    view-like and allocates the whole thing, so the gate is spread that way.
    """
    mlir = _forward_mlir()
    kda_chunk_lines = [
        line for line in mlir.splitlines() if "kda_chunk" in line
    ]
    assert kda_chunk_lines, "no kda_chunk op in the graph to check"
    assert "mo.broadcast_to" not in mlir, (
        "the gate spread regressed to a broadcast view; a TMA descriptor over "
        "the flat gate would read past a buffer 1/K the size"
    )
