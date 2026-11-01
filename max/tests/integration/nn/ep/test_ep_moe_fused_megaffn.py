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
"""EP8 runtime test for the one-launch fused EP MoE (``mega_ffn.ep_fused``).

Geometry is GLM-5.3's MoE (hidden 6144, 256 routed experts, MoE 2048, top-k 8)
at EP8 (32 local experts) with a decode capacity of 24 tokens per
rank, in both formats: NVFP4 with GLM-5.3's BF16 shared expert, and MXFP8
routed-only. The gates are `fused_megaffn_gates`'s; this module runs them on
devices 0-7:

* the shipping reference alone, one and two layers per graph, finite and
  repeatable under changing generations;
* one fused arm, eager or captured, against the shipping reference, which
  must not change after the fused arm ran (a bisect aid);
* production allocation and lowering; two chained layers as distinct module
  instances on one workspace; two independent EP instances; CUDA-graph
  capture replayed across changing generations with a NaN canary, against the
  eager one-layer shipping reference executed layer by layer; the host plan
  query equal to what EP init allocated, at token block 8;
* a batch of 25 tokens per rank, refused on the host before any launch;
* a capacity above 24 tokens per rank refused at EP init with a logged
  reason, the graph keeping the shipping chain; the allreduce backend refused
  at EP init;
* the fail-closed host checks: no workspace, or one of another layout;
* a fused and a shipping model alternating on one EP instance;
* the captured launch trace equal to eager execution's;
* an nsys subject (``EP8_FUSED_NSYS=1``), whose profile must show one fused
  kernel per layer and device per replay;
* one real GLM-5.3 NVFP4 MoE layer (its 256 routed experts with their real
  E4M3 block scales, per-tensor ``weight_scale_2`` and ``input_scale``, its
  BF16 shared expert and its router matrix) from a local GLM-5.3 NVFP4
  checkpoint named by ``GLM53_NVFP4_SNAPSHOT`` (layer
  ``GLM53_LAYER``, default 10), through the replay gate. Routing there uses
  the generic top-k gate on the real router matrix (not GLM's sigmoid +
  correction-bias router), and the hidden states are synthetic; both arms see
  the same ones.

Synthetic weights and routing elsewhere: a runtime-integration gate, not a
model-quality test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from fused_megaffn_gates import (
    FORMATS,
    Geometry,
    graph_replay,
    launch_fails_closed,
    mode_vs_ship,
    nsys_subject,
    refused_at_ep_init,
    refused_capacity,
    replay_trace_matches,
    shared_instance,
    ship_arrival_count_refused,
    ship_reference_repeats,
    wrap,
)
from max.driver import accelerator_api, accelerator_count
from max.graph.weights import WeightData
from safetensors import safe_open
from test_common.graph_utils import is_b100_b200

GEO = Geometry(n_devices=8, num_experts=256, max_tokens_per_rank=24)

# Source tokens per rank: the balanced decode batches across the capacity,
# the ragged one with a light rank, the skewed verify step, and the batch-1
# MTP verify step (six ranks hold one token, two hold none).
CELLS = {
    "dense8": [8] * 8,
    "dense16": [16] * 8,
    "dense24": [24] * 8,
    "ragged5": [24] * 7 + [5],
    "skew": [1, 3, 24, 5, 7, 12, 2, 24],
    "bs1_q6": [1, 1, 1, 1, 1, 1, 0, 0],
}

# The capacity rule EP init applies above the decode block's 24 tokens.
ABOVE_DECODE_REASON = "gated at 2 ranks only"


def _glm_layer_weights(
    snapshot: Path, layer: int
) -> dict[str, WeightData | torch.Tensor]:
    """One real GLM-5.3 NVFP4 MoE layer, renamed to the MoE module's keys."""
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.layers.{layer}.mlp."
    names = [n for n in index if n.startswith(prefix)]
    assert names, f"no MoE tensors for layer {layer} in {snapshot}"
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    w: dict[str, torch.Tensor] = {}
    for fname, tensors in by_file.items():
        with safe_open(str(snapshot / fname), framework="pt") as f:
            for n in tensors:
                key = n[len(prefix) :]
                if key == "gate.weight":
                    key = "gate.gate_score.weight"
                elif key.startswith("gate."):
                    continue  # e_score_correction_bias: unused by MoEGate
                w[key] = f.get_tensor(n)
    return wrap(w)


_needs_ep8 = [
    pytest.mark.skipif(
        accelerator_api() == "hip", reason="NVIDIA-only kernels"
    ),
    pytest.mark.skipif(not is_b100_b200(), reason="requires B100/B200 (SM100)"),
    pytest.mark.skipif(
        accelerator_count() < GEO.n_devices, reason="requires 8 GPUs (EP8)"
    ),
]


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize(
    ("fmt", "layers"),
    [("nvfp4", 1), ("mxfp8", 1), ("mxfp8", 2), ("nvfp4", 2)],
)
def test_ep8_ship_reference_repeats(fmt: str, layers: int) -> None:
    """The shipping reference alone, healthy under repeated execution.

    With two layers the graph holds two side-stream regions, one per layer's
    shared expert, so each layer needs its own module: a sharded weight
    materializes its shard on first use and caches it (`Weight._mlir_value`),
    and a module reused for the second layer would hand that region the
    first region's values, which fails graph verification ("operand #0 does
    not dominate this use" on the shared expert's gate/up concat).
    """
    ship_reference_repeats(GEO, fmt, layers, CELLS)


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize(
    "words",
    [
        # One slot short of 32 local experts + 1.
        32 * 32,
        # Not a whole number of 32-word slots (the allocator's 225 slots, less
        # one word).
        225 * 32 - 1,
    ],
)
def test_ep8_ship_arrival_count_undersized_refused(words: int) -> None:
    """The NVFP4 combine send refuses an ``arrival_count`` that cannot hold
    one 32-word slot per local expert plus one. The allocator's own size, 225
    slots at 24 tokens per rank, runs in `test_ep8_ship_reference_repeats`.
    """
    ship_arrival_count_refused(GEO, words, CELLS["dense24"])


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize("mode", ("eager", "graph"))
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_mode_vs_ship(fmt: str, mode: str) -> None:
    mode_vs_ship(GEO, fmt, mode, CELLS)


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_graph_replay(fmt: str) -> None:
    # Capacity 24 keeps the decode specialization (token block 8).
    assert graph_replay(GEO, fmt, CELLS) == 8


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_replay_trace_matches(fmt: str) -> None:
    replay_trace_matches(GEO, fmt, CELLS["dense24"])


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize("max_tpr", (25, 32))
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_refused_capacity(fmt: str, max_tpr: int) -> None:
    """Above 24 tokens per rank the fused path is gated at 2 ranks only: at
    EP8 EP init must refuse, log why, allocate nothing, and the layers keep
    the shipping chain."""
    refused_capacity(GEO, fmt, max_tpr, ABOVE_DECODE_REASON)


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
def test_ep8_fused_refused_under_allreduce() -> None:
    """The allreduce backend's EP ops see one rank, which the backend's rank
    rule refuses before it allocates."""
    refused_at_ep_init(GEO, "nvfp4", "allreduce", use_allreduce=True)


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no_workspace", "no fused EP MoE workspace"),
        ("other_layout", "set up for another configuration"),
    ],
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_launch_fails_closed(fmt: str, case: str, match: str) -> None:
    launch_fails_closed(GEO, fmt, case, match, smaller_capacity=16)


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_and_shipping_share_an_ep_instance(fmt: str) -> None:
    shared_instance(
        GEO, fmt, {c: CELLS[c] for c in ("dense24", "skew", "bs1_q6")}
    )


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.skipif(
    os.environ.get("EP8_FUSED_NSYS", "0") != "1",
    reason="nsys subject: set EP8_FUSED_NSYS=1",
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep8_fused_nsys_subject(fmt: str) -> None:
    nsys_subject(
        GEO,
        fmt,
        CELLS["dense24"],
        int(os.environ.get("EP8_FUSED_NSYS_REPLAYS", "3")),
    )


@_needs_ep8[0]
@_needs_ep8[1]
@_needs_ep8[2]
@pytest.mark.skipif(
    "GLM53_NVFP4_SNAPSHOT" not in os.environ,
    reason="set GLM53_NVFP4_SNAPSHOT to a local RadixArk/GLM-5.3-NVFP4 snapshot",
)
def test_ep8_fused_real_glm_layer() -> None:
    snapshot = Path(os.environ["GLM53_NVFP4_SNAPSHOT"])
    layer = int(os.environ.get("GLM53_LAYER", "10"))
    assert (
        graph_replay(
            GEO, "nvfp4", CELLS, wrapped=_glm_layer_weights(snapshot, layer)
        )
        == 8
    )
