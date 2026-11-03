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
"""A top-6 MoE cell as one fused EP launch
(``mega_ffn.ep_fused``).

The cell is production's per-rank geometry: 96 local experts (192 routed over
2 ranks), top-6, hidden 5120, MoE width 2304, NVFP4, at a 128-token per-rank
capacity, through the production op path with `fused_megaffn_gates`'s
gates: CUDA-graph capture replayed across generations against the eager
shipping reference, a fused and a shipping model alternating on one EP
instance, and the host plan. Its neighbors keep the chain: the same shape at
top-k 8, one token of capacity more, and 384 routed experts
over 2 ranks (192 local, past the scheduler's 128).

The production EP4 layout (384 routed experts, 96 local) is the same per-rank
cell at 4 ranks; these devices are two.
"""

from __future__ import annotations

import dataclasses

import pytest
from fused_megaffn_gates import (
    Geometry,
    ep_config,
    graph_replay,
    planned_as_refused,
    shared_instance,
)
from max.driver import accelerator_api, accelerator_count
from max.nn.legacy.comm.ep.ep_manager import _plan_fused_moe_workspace
from test_common.graph_utils import is_b100_b200

GEO = Geometry(
    n_devices=2,
    num_experts=192,
    max_tokens_per_rank=128,
    hidden_dim=5120,
    moe_dim=2304,
    top_k=6,
    n_layers=1,
)

# ``bs1``: one rank holds no tokens at all.
CELLS = {
    "dec8": [8, 8],
    "dense48": [48, 48],
    "dense102": [102, 102],
    "ragged102_48": [102, 48],
    "bs1": [1, 0],
}

# The shape's refusals name these rules.
TOPK_RULE = "local experts"
CAPACITY_RULE = "capacity"

_needs_ep2 = [
    pytest.mark.skipif(
        accelerator_api() == "hip", reason="NVIDIA-only kernels"
    ),
    pytest.mark.skipif(not is_b100_b200(), reason="requires B100/B200 (SM100)"),
    pytest.mark.skipif(
        accelerator_count() < GEO.n_devices,
        reason=f"requires {GEO.n_devices} GPUs",
    ),
]


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", ["nvfp4"])
def test_top6_fused_graph_replay(fmt: str) -> None:
    """The fused cell against the shipping chain on the same inputs."""
    token_block = graph_replay(GEO, fmt, CELLS)
    assert token_block == 8, token_block


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", ["nvfp4"])
def test_top6_fused_shared_instance(fmt: str) -> None:
    """A fused and a shipping model alternate on ONE EP instance."""
    shared_instance(GEO, fmt, {"dec8": [8, 8], "dense102": [102, 102]})


@pytest.mark.parametrize("fmt", ["nvfp4"])
def test_top6_planned(fmt: str) -> None:
    """The host plan serves the cell and budgets its arena."""
    size, token_block, reason = _plan_fused_moe_workspace(
        ep_config(GEO, fmt, True, shared=False)
    )
    assert (token_block, reason) == (8, ""), reason
    assert size > 0
    print(f"top6 {fmt}: fused workspace {size} B per rank, block {token_block}")


def test_top6_neighbors_refused() -> None:
    """One dimension off the cell keeps the shipping chain."""
    assert TOPK_RULE in planned_as_refused(
        dataclasses.replace(GEO, top_k=8), "nvfp4", TOPK_RULE
    )
    assert CAPACITY_RULE in planned_as_refused(
        dataclasses.replace(GEO, max_tokens_per_rank=129),
        "nvfp4",
        CAPACITY_RULE,
    )
    assert TOPK_RULE in planned_as_refused(
        dataclasses.replace(GEO, num_experts=384), "nvfp4", TOPK_RULE
    )
