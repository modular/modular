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
"""EP2 runtime test for the one-launch fused EP MoE (``mega_ffn.ep_fused``)
above 32 local experts.

The MoE of hidden 6144, MoE width 2048 and top-k 8 with 128 and with 256
routed experts (64 and 128 local) on devices 0-1, in both formats, at the one
capacity the backend serves them with (32 tokens per rank, on the decode token
block), through the production op path with `fused_megaffn_gates`'s gates:
CUDA-graph capture replayed across generations against the eager shipping
reference, a fused and a shipping model alternating on one EP instance, a
batch past the capacity refused on the host, and the host plan; at 128 local
also a launch without a fused workspace. Their neighbors keep the shipping
chain: the plan query refuses each count at capacity 24, one local expert
more, and 128 local experts at 4 ranks, and EP init refuses the first two.
"""

from __future__ import annotations

import dataclasses

import pytest
from fused_megaffn_gates import (
    FORMATS,
    Geometry,
    ep_config,
    graph_replay,
    launch_fails_closed,
    planned_as_refused,
    refused_at_ep_init,
    refused_capacity,
    shared_instance,
)
from max.driver import accelerator_api, accelerator_count
from max.nn.comm.ep.ep_manager import (
    _bound_fused_moe_workspace,
    _plan_fused_moe_workspace,
)
from test_common.graph_utils import is_b100_b200

GEO = Geometry(n_devices=2, num_experts=256, max_tokens_per_rank=32)

# ``bs1``: one rank holds no tokens at all.
CELLS = {
    "dense32": [32, 32],
    "dense24": [24, 24],
    "ragged": [24, 5],
    "bs1": [1, 0],
}

# The backend's refusal above 32 local experts names this rule.
LOCAL_RULE = "local experts"

# The admitted geometries above 32 local experts, by local expert count.
GEOS = {
    "local64": dataclasses.replace(GEO, num_experts=128),
    "local128": GEO,
}


def _over_local_experts(geo: Geometry) -> int:
    """Routed experts with one local expert more than ``geo``."""
    return geo.n_devices * (geo.num_experts // geo.n_devices + 1)


_needs_ep2 = [
    pytest.mark.skipif(
        accelerator_api() == "hip", reason="NVIDIA-only kernels"
    ),
    pytest.mark.skipif(not is_b100_b200(), reason="requires B100/B200 (SM100)"),
    pytest.mark.skipif(
        accelerator_count() < GEO.n_devices, reason="requires 2 GPUs"
    ),
]


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("geo", GEOS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_wide_fused_graph_replay(fmt: str, geo: str) -> None:
    assert graph_replay(GEOS[geo], fmt, CELLS) == 8


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("geo", GEOS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_wide_fused_shared_instance(fmt: str, geo: str) -> None:
    shared_instance(GEOS[geo], fmt, {"dense32": CELLS["dense32"]})


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_wide_fused_launch_without_workspace(fmt: str) -> None:
    # A workspace of another layout cannot be built here: every other capacity
    # at 128 local experts is refused when the graph compiles. The EP2 and EP8
    # tests at 32 or fewer local experts cover that host check.
    launch_fails_closed(
        GEO,
        fmt,
        "no_workspace",
        "no fused EP MoE workspace",
        smaller_capacity=GEO.max_tokens_per_rank,
    )


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("geo", GEOS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_wide_fused_refused_capacity(fmt: str, geo: str) -> None:
    g = GEOS[geo]
    # The capacity is part of the admitted configuration, so the rule that
    # names it refuses a larger one.
    refused_capacity(g, fmt, g.max_tokens_per_rank + 1, LOCAL_RULE)


@pytest.mark.parametrize("geo", GEOS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_wide_planned(fmt: str, geo: str) -> None:
    """The host plan query serves the geometry at capacity 32 on the decode
    token block and the memory planner's bound covers it; it refuses the same
    experts at capacity 24, 128 local experts at 4 ranks, and one local
    expert more."""
    g = GEOS[geo]
    config = ep_config(g, fmt, True, shared=False)
    size, block, reason = _plan_fused_moe_workspace(config)
    assert (block, reason) == (8, ""), (block, reason)
    bound, bound_reason = _bound_fused_moe_workspace(
        config, config.dispatch_dtype
    )
    assert bound >= size > 0, (bound, size)
    assert bound_reason == "", bound_reason
    planned_as_refused(
        dataclasses.replace(g, max_tokens_per_rank=24), fmt, LOCAL_RULE
    )
    planned_as_refused(g, fmt, LOCAL_RULE, num_experts=_over_local_experts(g))
    planned_as_refused(
        dataclasses.replace(g, n_devices=4, max_tokens_per_rank=24),
        fmt,
        LOCAL_RULE,
        num_experts=4 * 128,
    )


@_needs_ep2[0]
@_needs_ep2[1]
@_needs_ep2[2]
@pytest.mark.parametrize("geo", GEOS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep2_wide_neighbors_refused_at_ep_init(fmt: str, geo: str) -> None:
    g = GEOS[geo]
    refused_at_ep_init(g, fmt, LOCAL_RULE, num_experts=_over_local_experts(g))
    refused_at_ep_init(
        dataclasses.replace(g, max_tokens_per_rank=24), fmt, LOCAL_RULE
    )
