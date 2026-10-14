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

"""DSpark: V4's speculative head. Reference: ``inference/model.py``
``DSparkAttention``, ``DSparkMarkovHead``.

Three MTP stages drafting ``dspark_block_size`` tokens at a time. What makes
them stages rather than three copies is that only the ends are special --
``mtp.0`` owns the projection that brings the main trunk's hidden states in,
``mtp.2`` owns everything on the way out, and ``mtp.1`` has no DSpark-specific
parameter at all.

The stages run in two phases over ragged batches, the shape a block
spec-decode driver hands them:

* **materialize** -- after a trunk forward over ``T`` tokens, each stage writes
  the latent of the trunk's projected hidden state for every one of them into
  its window (:meth:`DSparkAttention.prefill_cache`). Rows past what the
  iteration commits are written too and overwritten at the same positions by
  the next one; nothing reads them in between, because a block reads the
  window strictly below its own start.
* **block** -- ``b`` requests of ``K`` draft tokens each, request ``r``'s block
  at positions ``L_r .. L_r + K - 1`` where ``L_r`` is one past the last token
  the trunk committed (:meth:`DSparkAttention.decode`).

A stage's sliding window lives in the shared window leaf, at layer
``num_hidden_layers + stage``: DSpark stages assert ``compress_ratio == 0``, so
none of the CSA machinery applies to them.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.embedding import Embedding
from max.nn.layer import Module
from max.nn.linear import Linear

from ..model_config import DeepseekV4Config
from .attention import DeepseekV4Attention, weightless_rms_normalize
from .cache import KEY, DeepseekV4Cache, arange, scalar
from .ragged import RaggedRows
from .rope import apply_rope_tail
from .sparse_attention import sparse_attention


class DSparkAttention(DeepseekV4Attention):
    """Attention whose keys come from the trunk and whose queries do not.

    This is the substantive difference from the main block, more than the
    absent compressor: the window holds latents projected from the trunk's
    hidden state, while ``q`` comes from the draft tokens. The draft is
    querying the real sequence, not itself.
    """

    def prefill_cache(
        self, main_x: TensorValue, rows: RaggedRows, cache: DeepseekV4Cache
    ) -> None:
        """Write the latents of a ragged batch's trunk states to the window.

        Args:
            main_x: ``[1, T, hidden]``, the projected trunk state of every
                token of the trunk forward ``rows`` describes.
            rows: That forward's token bookkeeping; the window leaf's
                ``cache_lengths`` must be its ``starts``.
            cache: The paged leaves.
        """
        freqs_cis = ops.gather(
            self.rope.freqs_cis_base(), rows.positions, axis=0
        )
        kv = self.latent(main_x, freqs_cis)
        cache.swa.store(
            self.swa_layer,
            KEY,
            ops.reshape(kv, [rows.total, self.head_dim]),
            rows.offsets,
        )

    def decode(
        self,
        x: TensorValue,
        rows: RaggedRows,
        cache: DeepseekV4Cache,
        block_size: int,
    ) -> TensorValue:
        """One draft block per request against the stage's window.

        Args:
            x: ``[1, b * block_size, hidden]`` draft tokens, already normed.
            rows: The block's bookkeeping: ``block_size`` rows per request,
                request ``r``'s ``starts`` being ``L_r``, one past the last
                committed token.
            cache: The paged leaves; the window already holds positions
                ``< L_r``.
            block_size: ``K``, the rows per request.

        Returns:
            ``[1, b * block_size, hidden]``.

        ``get_dspark_topk_idxs`` in the reference ranks nothing despite its
        name: every draft position sees the ``window`` positions below
        ``L_r`` and then the block's own rows, and the block is *not* causally
        masked within itself -- position 0 of the block attends to position 4.
        That is the reference's behaviour, and it is consistent with the block
        being drafted in one shot rather than autoregressively. The block's
        own latents are never stored.
        """
        rd = self.rope_head_dim
        t = rows.total
        device = x.device
        window = self.window
        freqs_cis = ops.gather(
            self.rope.freqs_cis_base(), rows.positions, axis=0
        )

        qr = self.q_norm(self.wq_a(x))
        q = ops.reshape(self.wq_b(qr), [1, t, self.n_heads, self.head_dim])
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)
        kv = self.latent(x, freqs_cis)

        # Each request's live window, positions ``L - window .. L - 1``; the
        # slots below position 0 are clamped onto a live one and masked.
        back = ops.reshape(arange(window, device) - window, [1, window])
        win_pos = ops.reshape(rows.starts, [rows.batch, 1]) + back
        zero = scalar(0, device)
        live = cache.swa.gather(self.swa_layer, KEY, ops.max(win_pos, zero))
        # Table: the block's rows first, so a token's block rows are addressed
        # by the batch's own row numbers, then ``window`` rows per request.
        table = ops.concat(
            [kv, ops.reshape(live, [1, rows.batch * window, self.head_dim])],
            axis=1,
        )
        tok_win_pos = ops.gather(win_pos, rows.bid, axis=0)
        win_rows = (
            rows.end
            + ops.reshape(rows.bid, [t, 1]) * scalar(window, device)
            + ops.reshape(arange(window, device), [1, window])
        )
        block_first = ops.reshape(rows.index - rows.local, [t, 1])
        idxs = ops.concat(
            [
                ops.where(tok_win_pos >= zero, win_rows, scalar(-1, device)),
                block_first
                + ops.reshape(arange(block_size, device), [1, block_size]),
            ],
            axis=-1,
        )
        o = sparse_attention(
            q, table, self.attn_sink, ops.unsqueeze(idxs, 0), self.softmax_scale
        )
        o = apply_rope_tail(o, freqs_cis, rd, inverse=True)
        return self._output_projection(o)


class DSparkMarkovHead(Module):
    """A rank-``dspark_markov_rank`` bigram correction on top of the logits.

    ``markov_w1`` is an embedding over the vocabulary and ``markov_w2`` is a
    projection back out to it, so the pair is a low-rank vocab-to-vocab map:
    given the token just emitted, it says which tokens become more or less
    likely next. Both are ``[vocab, rank]`` and they are *not* interchangeable
    -- one is looked up, the other is a matmul weight.

    Its output is added to the logits. It does not replace or gate them.
    """

    def __init__(self, config: DeepseekV4Config, device: DeviceRef) -> None:
        super().__init__()
        self.markov_w1 = Embedding(
            config.vocab_size,
            config.dspark_markov_rank,
            dtype=config.dtype,
            device=device,
        )
        self.markov_w2 = Linear(
            config.dspark_markov_rank,
            config.vocab_size,
            config.dtype,
            device,
        )

    def projection(self) -> TensorValue:
        """``markov_w2`` as the float32 ``[rank, vocab]`` matmul operand.

        ParallelHead computes logits in float32 off a bf16 weight. Built once
        per block, not once per Markov step.
        """
        return ops.transpose(
            ops.cast(self.markov_w2.weight, DType.float32), 0, 1
        )

    def __call__(
        self, token_ids: TensorValue, projection: TensorValue
    ) -> TensorValue:
        """``[b]`` token ids -> ``[b, vocab]`` float32 logit bias."""
        return ops.matmul(
            ops.cast(self.markov_w1(token_ids), DType.float32), projection
        )
