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
"""EP4 runtime test for the one-launch fused EP MoE (``mega_ffn.ep_fused``)
above 32 local experts.

GLM-5.3's 256 routed experts on devices 0-3 (64 local), hidden 6144, MoE 2048,
top-k 8 and a capacity of 24 tokens per rank, in both formats (MXFP8 on the
decode token block, NVFP4 on token block 32), through the production op path
with `fused_megaffn_gates`'s gates: a balanced and a ragged cell (with an
empty rank) captured once and replayed across generations against the eager
shipping reference, a batch past the capacity refused before any launch, and
a capacity above 24 refused at EP init. The host plan and the neighbors are in
`test_ep_moe_fused_megaffn_ep4.py`.
"""

from __future__ import annotations

import pytest
from fused_megaffn_gates import (
    FORMATS,
    Geometry,
    graph_replay,
    refused_capacity,
)
from max.driver import accelerator_api, accelerator_count
from test_common.graph_utils import is_b100_b200

GEO = Geometry(n_devices=4, num_experts=256, max_tokens_per_rank=24)

CELLS = {
    "balanced24": [24, 24, 24, 24],
    "ragged": [24, 5, 0, 13],
}

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
def test_ep4_wide_fused_graph_replay(fmt: str) -> None:
    # NVFP4 runs token block 32 here, MXFP8 the decode block.
    assert graph_replay(GEO, fmt, CELLS) == (32 if fmt == "nvfp4" else 8)


@_needs_ep4[0]
@_needs_ep4[1]
@_needs_ep4[2]
@pytest.mark.parametrize("fmt", FORMATS)
def test_ep4_wide_fused_refused_capacity(fmt: str) -> None:
    # The capacity is part of the admitted configuration, so the rule that
    # names it refuses a larger one.
    refused_capacity(GEO, fmt, GEO.max_tokens_per_rank + 1, "local experts")
