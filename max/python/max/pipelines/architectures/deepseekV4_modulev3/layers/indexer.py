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
"""The DeepSeek-V4 lightning indexer on ModuleV3, cached path only.

Scores each query against its own compressed zone leaf in place and keeps
the top ``index_topk`` closed entries. The query side (RoPE, Hadamard, FP4)
and the two causality rules are described in the ModuleV2 implementation
(``deepseekV4/layers/indexer.py`` at dfc8109fc0c).
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from ..model_config import DeepseekV4Config
from .compressor import DeepseekV4Compressor
from .csa import CompressedStream, int32
from .hadamard import hadamard_rotate
from .kernels import indexer_score_ragged, top_k_per_row
from .quantization import fp4_qat_quantize, linear_for, shard_linear
from .rope import apply_rope_tail

__all__ = ["DeepseekV4Indexer"]


class DeepseekV4Indexer(Module[..., Tensor]):
    """Picks the compressed entries a ``compress_ratio == 4`` query attends to."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        # Both softmax scales the reference folds into the head weights; the
        # head count is the full one (the reference divides by it).
        self.weights_scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.wq_b = linear_for(
            config, config.q_lora_rank, self.n_heads * self.head_dim
        )
        self.weights_proj = Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(
            config, compress_ratio, self.head_dim, rotate=True
        )

    def shard_heads(self) -> None:
        """``wq_b`` and ``weights_proj`` per head; the compressor stays whole."""
        n = self.weights_proj.weight.mesh.num_devices
        if self.n_heads % n:
            raise ValueError(
                f"{self.n_heads} indexer heads do not split across {n} devices"
            )
        shard_linear(self.wq_b, 0)
        shard_linear(self.weights_proj, 0)

    def _query(
        self, x: Tensor, qr: Tensor, freqs_cis: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``[T, heads, head_dim]`` queries and ``[T, heads]`` float32 weights."""
        t = x.shape[0]
        q = self.wq_b(qr).reshape([t, self.n_heads, self.head_dim])
        q = apply_rope_tail(q, freqs_cis, self.rope_head_dim)
        # Rotate, then quantize: the rotation is what makes FP4 tolerable.
        q = fp4_qat_quantize(hadamard_rotate(q))
        weights = self.weights_proj(x).cast(DType.float32) * self.weights_scale
        return q, weights

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        freqs_cis: Tensor,
        stream: CompressedStream,
        cutoff: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """``[T, k]`` int32 candidate numbers, ``-1`` in unusable slots.

        Args:
            x: ``[T, hidden]`` block input.
            qr: ``[T, q_lora_rank]``, the attention's ``q_norm(wq_a(x))``.
            freqs_cis: ``[T, rope_head_dim // 2, 2]`` rotary rows.
            stream: The indexer's own compressed stream, already stored.
            cutoff: ``[T]`` int32, ``(position + 1) // ratio`` per query.
            valid: ``[T, n_cand]`` bool, the attention's candidate mask.
        """
        q, weights = self._query(x, qr, freqs_cis)
        score = indexer_score_ragged(
            q,
            weights,
            stream.rows.offsets,
            stream.windows.base,
            cutoff,
            stream.zone_leaf.values,
            stream.layer,
            num_candidates=stream.n_cand,
        )
        # Summed over every device's heads, so all devices pick alike.
        score = score.to(DeviceMapping.replicated(score.mesh))
        return self.select(score, valid)

    def select(self, score: Tensor, valid: Tensor) -> Tensor:
        """The top-k of ``[T, n]`` scores under ``valid``."""
        n = int(score.shape[1])
        neg_inf = F.constant(float("-inf"), DType.float32, device=score.mesh)
        # Rule 1: an entry not closed by the query cannot be selected, nor a
        # NaN score, which the top-k does not accept.
        score = F.where(valid & ~F.is_nan(score), score, neg_inf)
        k = min(self.index_topk, n)
        # Until more than ``k`` entries have closed, select all of them.
        live = valid.cast(DType.int64).sum(axis=-1).squeeze(-1)
        picks = F.elementwise_min(
            live, F.constant(k, DType.int64, device=live.mesh)
        )
        top_scores, top_idxs = top_k_per_row(score, picks, k)
        # Rule 2: drop any pick whose score was not finite.
        return F.where(
            top_scores > neg_inf,
            top_idxs.cast(DType.int32),
            int32(-1, top_idxs),
        )
