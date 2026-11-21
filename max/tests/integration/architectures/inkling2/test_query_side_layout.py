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
"""Layout checks for the Inkling 2 fused q-side projection that compile no graph.

The numeric checks compile host graphs, minutes apiece on CI, so they run in
postsubmit (test_query_side_projection, test_query_side_shards); these run in
presubmit.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.inkling2.layers.attention import (
    _QuerySideProjection,
)
from max.tensor import Tensor, default_dtype
from numpy.typing import NDArray

HIDDEN = 16
HEADS = 8
HEAD_DIM = 4
D_REL = 2
WIDTH = 2 * HEAD_DIM + D_REL + 1
# Each head's rows round up to a multiple of 8, zero-filled.
PADDED_WIDTH = 16
TOKENS = 3
# The deployed degree: every rank's block has one shape, so it compiles once.
TP_DEGREES = [4]

Weights = dict[str, NDArray[np.float32]]


def _random_weights(rng: np.random.Generator) -> Weights:
    """Per-projection weights at the checkpoint's shapes."""

    def normal(*shape: int) -> NDArray[np.float32]:
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "wq_du": normal(HEADS * HEAD_DIM, HIDDEN),
        "wr_du": normal(HEADS * D_REL, HIDDEN),
        "gate_attn": normal(HEADS * HEAD_DIM, HIDDEN),
        "attn_sink_proj": normal(HEADS, HIDDEN),
    }


def _build(weights: Weights, num_heads: int) -> _QuerySideProjection:
    """Builds the projection holding ``num_heads`` heads' worth of weights."""
    with default_dtype(DType.float32), F.lazy():
        proj = _QuerySideProjection(
            hidden=HIDDEN,
            num_heads=num_heads,
            head_dim=HEAD_DIM,
            d_rel=D_REL,
        )
    for name, weight in weights.items():
        getattr(proj, name).weight = Tensor(weight, device=CPU())
    return proj


def test_projection_names_stay_native() -> None:
    """The fused layer loads the checkpoint's four projection names."""
    proj = _build(_random_weights(np.random.default_rng(0)), HEADS)
    names = {name for name, _ in proj.parameters}
    assert names == {
        "wq_du.weight",
        "wr_du.weight",
        "gate_attn.weight",
        "attn_sink_proj.weight",
    }


def test_inkling2_heads_pad_to_aligned_rows() -> None:
    """Inkling 2's 273 rows per head pad to 280, so 12 heads per rank align."""
    with F.lazy():
        proj = _QuerySideProjection(
            hidden=HIDDEN, num_heads=12, head_dim=128, d_rel=16
        )
    assert proj.width == 273
    assert proj.padded_width == 280
    # A rank's bf16 output row is 16-byte aligned, as the SM100 matmul needs.
    assert proj.num_heads * proj.padded_width * 2 % 16 == 0
