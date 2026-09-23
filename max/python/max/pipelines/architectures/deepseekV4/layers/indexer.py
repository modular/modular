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

"""The DeepSeek-V4 lightning indexer.

Reference: ``inference/model.py`` class ``Indexer``. Only ``compress_ratio == 4``
layers have one. It decides *which* compressed KV entries a query is allowed to
see; ``compress_ratio == 128`` layers skip it and take every entry that closed
before them.

It is a second, much narrower attention that scores queries against its own
compressed KV:

* 64 heads of width 128, against the main block's 128 heads of width 512.
* Its own ``Compressor`` over the same ``x``, at the same ratio, but built with
  ``rotate=True`` -- Hadamard rotation and FP4 instead of the main path's FP8.
  Its entries live in their own pair of cache leaves; the attention layer runs
  that stream and hands the indexer the candidate table.
* The query side gets the same treatment: RoPE on the trailing 64 dims, then
  Hadamard, then FP4. The rotation is what makes 4-bit survivable -- it spreads
  a single outlier dim across the whole 128-wide vector before the e2m1 grid,
  whose eight magnitudes would otherwise let that outlier set its block's scale
  alone.
* Scores are ``relu``'d, weighted per head by a learned ``weights_proj`` over
  the block input, and summed across heads into one score per (query, entry).
  Nothing is normalized -- these are ranking scores, never a distribution, which
  is why FP4 is enough.

Causality is enforced twice, and the two are not the same rule:

1. Before the top-k, entries that had not closed yet are ``-inf``'d so they
   cannot be selected.
2. After the top-k, any index that still points at a not-yet-closed entry
   becomes ``-1``. This fires when fewer than ``k`` entries are available: the
   top-k has to return ``k`` indices, so it pads out of the ``-inf`` region, and
   those picks are dropped here instead. ``-1`` is what ``sparse_attention``
   reads as "no such position".
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear

from ..model_config import DeepseekV4Config
from .compressor import DeepseekV4Compressor
from .hadamard import hadamard_rotate
from .quantization import fp4_qat_quantize
from .rope import apply_rope_tail


class DeepseekV4Indexer(Module):
    """Picks the compressed entries a ``compress_ratio == 4`` query attends to."""

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        # Note this is the *indexer's* head width, 128, not the block's 512.
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = Linear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            config.dtype,
            device,
        )
        # The reference pins this one to bf16 explicitly
        # (``dtype=torch.bfloat16`` on the ColumnParallelLinear), which for this
        # checkpoint is what ``config.dtype`` already is; ``wo_a`` in the block
        # is declared the same way and followed the config there too.
        self.weights_proj = Linear(
            config.hidden_size, self.n_heads, config.dtype, device
        )
        self.compressor = DeepseekV4Compressor(
            config, compress_ratio, self.head_dim, device, rotate=True
        )

    def __call__(
        self,
        x: TensorValue,
        qr: TensorValue,
        freqs_cis: TensorValue,
        candidates: TensorValue,
        valid: TensorValue,
    ) -> TensorValue:
        """Select compressed entries for every query in a ragged batch.

        Args:
            x: ``[T, hidden_size]``, the block input the main attention also
                consumes.
            qr: ``[T, q_lora_rank]``, the attention's ``q_norm(wq_a(x))``.
                Shared, not recomputed -- the indexer reads the same low-rank
                query the block does.
            freqs_cis: ``[T, rope_head_dim // 2, 2]``, the rotary rows at the
                tokens' positions.
            candidates: ``[T, n, index_head_dim]`` each token's candidate
                table: the indexer's own compressed entries in candidate order
                (closed entries from the cache, then its request's fresh
                windows).
            valid: ``[T, n]`` bool, which candidates each query may see -- the
                same mask the attention applies.

        Returns:
            ``[T, k]`` int32 candidate numbers (rows of ``candidates``), with
            ``-1`` in unusable slots. ``k = min(index_topk, n)``.
        """
        device = x.device
        t = x.shape[0]
        n = int(candidates.shape[1])

        # ``apply_rope_tail`` wants the sequence on axis 1.
        q = ops.reshape(self.wq_b(qr), [1, t, self.n_heads, self.head_dim])
        q = apply_rope_tail(q, freqs_cis, self.rope_head_dim)
        # Rotate first, then quantize: the rotation is there to make the FP4
        # grid tolerable, so the order is not interchangeable.
        q = fp4_qat_quantize(hadamard_rotate(q))

        q32 = ops.reshape(
            ops.cast(q, DType.float32), [t, self.n_heads, self.head_dim]
        )
        # "thd,tnd->thn": one rank-3 batched matmul with the tokens as the
        # batch. Rank 3 also keeps this off the batched-matmul path whose
        # fused epilogue (the relu) cannot instantiate for rank-4 outputs once
        # n is a multiple of 128 (ISSUES.md Issue 32).
        kv_t = ops.transpose(ops.cast(candidates, DType.float32), -1, -2)
        scores = ops.relu(ops.matmul(q32, kv_t))

        # One learned weight per head per query, folded with both softmax
        # scales the reference applies here rather than to the scores.
        weights = ops.cast(self.weights_proj(x), DType.float32) * (
            self.softmax_scale * self.n_heads**-0.5
        )
        index_score = ops.squeeze(
            ops.sum(scores * ops.unsqueeze(weights, -1), axis=1), axis=1
        )

        # Rule 1: an entry that had not closed by the query cannot be selected.
        index_score = ops.where(
            valid,
            index_score,
            ops.constant(float("-inf"), DType.float32, device),
        )

        k = min(self.index_topk, n)
        topk_scores, topk_idxs = ops.top_k(index_score, k, axis=-1)
        topk_idxs = ops.cast(topk_idxs, DType.int32)

        # Rule 2: drop the padding picks the top-k had to make out of the -inf
        # region.
        return ops.where(
            topk_scores > ops.constant(float("-inf"), DType.float32, device),
            topk_idxs,
            ops.constant(-1, DType.int32, device),
        )
