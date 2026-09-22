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
``DSparkAttention``, ``DSparkMarkovHead``, ``DSparkConfidenceHead``.

Three MTP stages drafting ``dspark_block_size`` tokens at a time. What makes
them stages rather than three copies is that only the ends are special --
``mtp.0`` owns the projection that brings the main trunk's hidden states in,
``mtp.2`` owns everything on the way out, and ``mtp.1`` has no DSpark-specific
parameter at all.

**The whole path is decode-only.** ``DSparkBlock.forward`` at ``start_pos == 0``
calls nothing but ``self.attn``, and ``DSparkAttention.forward`` at
``start_pos == 0`` fills its KV cache and returns its input untouched;
``Transformer.forward_spec`` then returns before producing logits. So DSpark
contributes nothing to a prefill, and nothing here can be checked against a
step-0 golden.

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
from .attention import (
    DeepseekV4Attention,
    chunk_positions,
    weightless_rms_normalize,
    window_table,
)
from .cache import KEY, DeepseekV4Cache, arange, row_offsets
from .rope import apply_rope_tail
from .sparse_attention import sparse_attention


class DSparkAttention(DeepseekV4Attention):
    """Attention whose keys come from the trunk and whose queries do not.

    This is the substantive difference from the main block, more than the
    absent compressor: ``main_kv`` is projected from ``main_x``, the trunk's
    hidden state, while ``q`` comes from the draft tokens. The draft is
    querying the real sequence, not itself.
    """

    def prefill_cache(
        self, main_x: TensorValue, seq_len: int, cache: DeepseekV4Cache
    ) -> None:
        """Write the latent rows a prefill leaves in the stage's window.

        ``DSparkAttention.forward`` at ``start_pos == 0`` computes exactly
        these rows and returns its input unchanged.
        """
        b = int(main_x.shape[0])
        positions = chunk_positions(
            cache.cache_lengths, b, seq_len, main_x.device
        )
        freqs_cis = ops.gather(self.rope.freqs_cis_base(), positions, axis=0)
        kv = self.latent(main_x, freqs_cis)
        cache.swa.store(
            self.swa_layer,
            KEY,
            ops.reshape(kv, [b * seq_len, self.head_dim]),
            row_offsets(b, seq_len, main_x.device),
        )

    def decode(
        self,
        x: TensorValue,
        main_x: TensorValue,
        cache: DeepseekV4Cache,
    ) -> TensorValue:
        """One draft block against the stage's window.

        Args:
            x: ``[b, block_size, hidden]`` draft tokens, already normed.
            main_x: ``[b, 1, hidden]`` the trunk's projected hidden state for
                the token at ``cache.cache_lengths``; its latent is written to
                the window before the block reads it.
            cache: The paged leaves.

        Returns:
            ``[b, block_size, hidden]``.

        ``get_dspark_topk_idxs`` in the reference ranks nothing despite its
        name: every draft position sees the whole live window and then the
        block's own rows, and the block is *not* causally masked within
        itself -- position 0 of the block attends to position 4. That is the
        reference's behaviour, and it is consistent with the block being
        drafted in one shot rather than autoregressively.
        """
        rd = self.rope_head_dim
        b = int(x.shape[0])
        block_size = int(x.shape[1])
        device = x.device
        table = self.rope.freqs_cis_base()

        # The trunk's token sits at ``P``; the draft block follows it.
        main_pos = chunk_positions(cache.cache_lengths, b, 1, device)
        main_kv = self.latent(main_x, ops.gather(table, main_pos, axis=0))
        cache.swa.store(
            self.swa_layer,
            KEY,
            ops.reshape(main_kv, [b, self.head_dim]),
            row_offsets(b, 1, device),
        )
        draft_pos = main_pos + ops.reshape(
            arange(block_size, device) + 1, [1, block_size]
        )
        freqs_cis = ops.gather(table, draft_pos, axis=0)

        qr = self.q_norm(self.wq_a(x))
        q = ops.reshape(
            self.wq_b(qr), [b, block_size, self.n_heads, self.head_dim]
        )
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)
        kv = self.latent(x, freqs_cis)

        # The window table is built for a "chunk" of the one trunk token, so
        # it holds positions ``P - window + 1 .. P``: exactly the live ring.
        window, win_idxs = window_table(
            cache.swa, self.swa_layer, main_kv, main_pos, self.window
        )
        live = ops.broadcast_to(win_idxs, [b, block_size, self.window])
        block_rows = ops.reshape(
            arange(block_size, device) + self.window, [1, 1, block_size]
        )
        idxs = ops.concat(
            [live, ops.broadcast_to(block_rows, [b, block_size, block_size])],
            axis=-1,
        )
        o = sparse_attention(
            q,
            ops.concat([window, kv], axis=1),
            self.attn_sink,
            idxs,
            self.softmax_scale,
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

    def __call__(
        self, token_ids: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """``[b]`` token ids -> ``([b, vocab]`` logit bias, ``[b, rank]`` embed)."""
        embed = self.markov_w1(token_ids)
        # ParallelHead computes logits in float32 off a bf16 weight.
        logits = ops.matmul(
            ops.cast(embed, DType.float32),
            ops.transpose(ops.cast(self.markov_w2.weight, DType.float32), 0, 1),
        )
        return logits, embed


class DSparkConfidenceHead(Module):
    """Scalar per draft position: how much to trust this draft.

    Reads the state *after* the mHC contraction but *before* the final norm --
    the reference does not reassign ``x`` between the two -- concatenated with
    the Markov embedding for the token at that position.
    """

    def __init__(self, config: DeepseekV4Config, device: DeviceRef) -> None:
        super().__init__()
        self.proj = Linear(
            config.hidden_size + config.dspark_markov_rank,
            1,
            DType.float32,
            device,
        )

    def __call__(
        self, hidden: TensorValue, markov_embed: TensorValue
    ) -> TensorValue:
        joined = ops.concat(
            [
                ops.cast(hidden, DType.float32),
                ops.cast(markov_embed, DType.float32),
            ],
            axis=-1,
        )
        return ops.squeeze(self.proj(joined), axis=-1)
