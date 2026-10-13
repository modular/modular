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
"""EP4 runtime test for the one-launch fused EP MoE (``mega_ffn.ep_fused``).

Four ranks on devices 0-3, 32 routed experts (8 per rank), hidden 6144, MoE
2048, top-k 8 and a capacity of 24 tokens per rank, in both formats, through
the production op path with `fused_megaffn_gates`'s gates: a balanced and a
ragged cell (with an empty rank) captured once and replayed across
generations against the eager shipping reference, a batch past the capacity
refused before any launch, and a capacity above 24 refused at EP init.

GLM-5.3's 256 routed experts at EP4 are 64 local experts, past the 32 the
fused path scans: EP init must refuse them and the memory planner must budget
nothing for them. That case checks the refusal only.
"""

from __future__ import annotations

import pytest
from fused_megaffn_gates import (
    FORMATS,
    Geometry,
    ep_config,
    graph_replay,
    refused_at_ep_init,
    refused_capacity,
)
from max.driver import accelerator_api, accelerator_count
from max.nn.comm.ep.ep_manager import (
    _bound_fused_moe_workspace,
    _plan_fused_moe_workspace,
)
from test_common.graph_utils import is_b100_b200

GEO = Geometry(n_devices=4, num_experts=32, max_tokens_per_rank=24)

CELLS = {
    "balanced24": [24, 24, 24, 24],
    "ragged": [24, 5, 0, 13],
}

GLM_ROUTED_EXPERTS = 256

_needs_ep4 = [
    pytest.mark.skipif(
        accelerator_api() == "hip", reason="NVIDIA-only kernels"
    ),
    pytest.mark.skipif(not is_b100_b200(), reason="requires B100/B200 (SM100)"),
    pytest.mark.skipif(
        accelerator_count() < GEO.n_devices, reason="requires 4 GPUs (EP4)"
    ),
]


@_needs_ep4[0]
@_needs_ep4[1]
@_needs_ep4[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep4_fused_graph_replay(fmt: str) -> None:
    # Capacity 24 keeps the decode specialization (token block 8).
    assert graph_replay(GEO, fmt, CELLS) == 8


@_needs_ep4[0]
@_needs_ep4[1]
@_needs_ep4[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep4_fused_refused_capacity(fmt: str) -> None:
    refused_capacity(
        GEO, fmt, GEO.max_tokens_per_rank + 1, "gated at 2 ranks only"
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_ep4_glm_expert_count_planned_as_refused(fmt: str) -> None:
    """The host plan query, exact and as the memory planner's bound, budgets
    no fused workspace for 64 local experts and names the local-expert
    rule."""
    config = ep_config(
        GEO, fmt, True, shared=False, num_experts=GLM_ROUTED_EXPERTS
    )
    size, token_block, reason = _plan_fused_moe_workspace(config)
    assert (size, token_block) == (0, 0)
    assert "local experts" in reason, reason
    bound, bound_reason = _bound_fused_moe_workspace(
        config, config.dispatch_dtype
    )
    assert bound == 0
    assert bound_reason == reason, (bound_reason, reason)


@_needs_ep4[0]
@_needs_ep4[1]
@_needs_ep4[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep4_glm_expert_count_refused_at_ep_init(fmt: str) -> None:
    """EP init on devices 0-3 for 256 routed experts refuses the fused path,
    logs why, allocates no workspace and leaves readiness off, so every MoE
    layer keeps the shipping chain."""
    reason = refused_at_ep_init(
        GEO, fmt, "local experts", num_experts=GLM_ROUTED_EXPERTS
    )
    print(f"{fmt}: 256 experts at EP4 refused at EP init ({reason})")
