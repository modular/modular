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
"""Inkling 2's block attention residual.

Every sublayer reads a softmax mixture of the earlier blocks' stored sums and
the live partial sum, in place of the plain residual stream. The mixture is
position-local: it holds no state across steps.
"""

from __future__ import annotations

from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.tensor import Tensor

from .norm import unweighted_rms_norm

ATTN_RES_EPS = 1e-6


class AttnResQueries(Module[[], None]):
    """The learned queries, under the checkpoint's names.

    ``attn_queries[i]`` feeds the attention of layer ``i + 1``: ``[3, hidden]``
    on a global layer, whose q, k and v each get their own mixture, else
    ``[hidden]``. ``mlp_queries[i]`` feeds the MLP of layer ``i``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        is_local: Sequence[bool],
    ) -> None:
        self.attn_queries = [
            Tensor.zeros(
                [hidden_size] if is_local[layer] else [3, hidden_size],
                dtype=DType.float32,
            )
            for layer in range(1, num_layers)
        ]
        self.mlp_queries = [
            Tensor.zeros([hidden_size], dtype=DType.float32)
            for _ in range(num_layers)
        ]
        self.output_query = Tensor.zeros([hidden_size], dtype=DType.float32)


def attn_res_mix(
    sources: Sequence[Tensor], queries: Tensor, *, normalize_values: bool
) -> list[Tensor]:
    """Mixes the sources once for each query row, in float32.

    Scores come from the RMS-normalized sources. The mixed values are the
    normalized sources too when ``normalize_values``.

    Args:
        sources: The stored blocks then the live partial sum, each
            ``[tokens, hidden]``.
        queries: ``[hidden]`` or ``[num_queries, hidden]`` float32.
        normalize_values: Whether the mixture averages the normalized sources.

    Returns:
        One float32 ``[tokens, hidden]`` mixture per query row.
    """
    hidden = int(sources[0].shape[-1])
    queries = queries.reshape([-1, hidden])
    stack = F.stack([s.cast(DType.float32) for s in sources], axis=1)
    normed = unweighted_rms_norm(stack, ATTN_RES_EPS)
    values = normed if normalize_values else stack
    mixes = []
    for row in range(int(queries.shape[0])):
        scores = (normed * queries[row]).sum(axis=-1) * hidden**-0.5
        probs = F.softmax(scores, axis=1)
        mixes.append((probs * values).sum(axis=1).squeeze(1))
    return mixes
