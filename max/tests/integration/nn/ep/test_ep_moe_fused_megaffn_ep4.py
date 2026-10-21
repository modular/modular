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

GLM-5.3's 256 routed experts at EP4 (64 local) at capacity 24 are served in
both formats, MXFP8 on the decode token block and NVFP4 on token block 32;
`test_ep_moe_fused_megaffn_ep4_wide.py` replays them. Here: the host plan for
them, and their neighbors (65 or 128 local experts, top-k 4, MoE width 3072,
capacity 25) refused by the plan query and, at 128 local, at EP init.
"""

from __future__ import annotations

import dataclasses

import pytest
from fused_megaffn_gates import (
    FORMATS,
    Geometry,
    ep_config,
    graph_replay,
    planned_as_refused,
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

GLM = dataclasses.replace(GEO, num_experts=256)
LOCAL_RULE = "local experts"

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
def test_ep4_glm_planned(fmt: str) -> None:
    """The host plan query serves 64 local experts at capacity 24, MXFP8 on
    the decode token block and NVFP4 on 32, and the memory planner's bound
    covers it; one local expert more, 128 local experts, top-k 4, MoE width
    3072 and capacity 25 get no fused workspace and name the local-expert
    rule."""
    config = ep_config(GLM, fmt, True, shared=False)
    size, block, reason = _plan_fused_moe_workspace(config)
    assert (block, reason) == (32 if fmt == "nvfp4" else 8, ""), (block, reason)
    bound, bound_reason = _bound_fused_moe_workspace(
        config, config.dispatch_dtype
    )
    assert bound >= size > 0, (bound, size)
    assert bound_reason == "", bound_reason
    planned_as_refused(GLM, fmt, LOCAL_RULE, num_experts=4 * 65)
    planned_as_refused(GLM, fmt, LOCAL_RULE, num_experts=4 * 128)
    planned_as_refused(dataclasses.replace(GLM, top_k=4), fmt, LOCAL_RULE)
    planned_as_refused(dataclasses.replace(GLM, moe_dim=3072), fmt, LOCAL_RULE)
    planned_as_refused(
        dataclasses.replace(GLM, max_tokens_per_rank=25), fmt, LOCAL_RULE
    )


@_needs_ep4[0]
@_needs_ep4[1]
@_needs_ep4[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep4_glm_neighbor_refused_at_ep_init(fmt: str) -> None:
    """EP init on devices 0-3 for 512 routed experts (128 local) refuses the
    fused path, logs why, allocates no workspace and leaves readiness off, so
    every MoE layer keeps the shipping chain."""
    reason = refused_at_ep_init(GLM, fmt, LOCAL_RULE, num_experts=4 * 128)
    print(f"{fmt}: 512 experts at EP4 refused at EP init ({reason})")
