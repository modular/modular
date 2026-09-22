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

The ring buffer is taken as an argument rather than owned here, so the
arithmetic can be gated before the cache plumbing exists. It is a plain
``window_size``-wide window -- DSpark stages assert ``compress_ratio == 0``, so
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
from .quantization import fp8_qat_quantize
from .rope import apply_rope_tail
from .sparse_attention import sparse_attention


def dspark_kv_idxs(
    window: int, block_size: int, start_pos: int, device: DeviceRef
) -> TensorValue:
    """``[block_size, min(window, start_pos + 1) + block_size]`` fixed indices.

    ``get_dspark_topk_idxs``. The name in the reference is misleading: nothing
    is ranked. Every draft position sees the same thing -- the whole live part
    of the ring buffer, then the ``block_size`` rows of this draft block, which
    sit at offsets ``window ...`` because the block is concatenated after the
    full-width cache.

    Note the draft block is *not* causally masked within itself: position 0 of
    the block attends to position 4. That is the reference's behaviour, and it
    is consistent with the block being drafted in one shot rather than
    autoregressively.
    """
    live = min(window, start_pos + 1)
    history = ops.range(
        0, live, 1, out_dim=live, device=device, dtype=DType.int32
    )
    block = window + ops.range(
        0,
        block_size,
        1,
        out_dim=block_size,
        device=device,
        dtype=DType.int32,
    )
    row = ops.concat([history, block], axis=0)
    return ops.broadcast_to(
        ops.unsqueeze(row, 0), [block_size, live + block_size]
    )


class DSparkAttention(DeepseekV4Attention):
    """Attention whose keys come from the trunk and whose queries do not.

    This is the substantive difference from the main block, more than the
    absent compressor: ``main_kv`` is projected from ``main_x``, the trunk's
    hidden state, while ``q`` comes from the draft tokens. The draft is
    querying the real sequence, not itself.
    """

    def prefill_cache(self, main_x: TensorValue, seq_len: int) -> TensorValue:
        """The latent rows a prefill writes into the ring, ``[b, seq_len, d]``.

        ``DSparkAttention.forward`` at ``start_pos == 0`` computes exactly this
        and returns its input unchanged; where the rows land in the ring is the
        cache's problem, not this one's. Positions start at 0, so the layer's
        rotary table is read from the front.
        """
        rd = self.rope_head_dim
        freqs_cis = self.rope.freqs_cis_base()[:seq_len]
        kv = self.kv_norm(self.wkv(main_x))
        kv = apply_rope_tail(kv, freqs_cis, rd)
        # Positive bounds: see the note in rope.py and ISSUES Issue 34.
        w = int(kv.shape[-1])
        return ops.concat(
            [fp8_qat_quantize(kv[..., : w - rd]), kv[..., w - rd :]], axis=-1
        )

    def decode(
        self,
        x: TensorValue,
        main_x: TensorValue,
        kv_cache: TensorValue,
        start_pos: int,
        seq_len: int,
    ) -> TensorValue:
        """One draft block against the ring buffer.

        Args:
            x: ``[b, block_size, hidden]`` draft tokens, already normed.
            main_x: ``[b, seq_len, hidden]`` the trunk's projected hidden; at
                decode ``seq_len`` is 1.
            kv_cache: ``[b, window, head_dim]`` ring buffer, with this step's
                ``main_kv`` already written at ``start_pos % window``.
            start_pos: Position of the token the trunk just produced.
            seq_len: ``main_x``'s length.

        Returns:
            ``[b, block_size, hidden]``.
        """
        rd = self.rope_head_dim
        block_size = int(x.shape[1])
        table = self.rope.freqs_cis_base()
        # The draft block sits *after* the trunk's current token, so its
        # positions start at start_pos + seq_len, not at start_pos.
        freqs_cis = table[
            start_pos + seq_len : start_pos + seq_len + block_size
        ]

        qr = self.q_norm(self.wq_a(x))
        q = ops.reshape(
            self.wq_b(qr),
            [x.shape[0], block_size, self.n_heads, self.head_dim],
        )
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)

        kv = self.kv_norm(self.wkv(x))
        kv = apply_rope_tail(kv, freqs_cis, rd)
        kv_width = int(kv.shape[-1])
        kv = ops.concat(
            [
                fp8_qat_quantize(kv[..., : kv_width - rd]),
                kv[..., kv_width - rd :],
            ],
            axis=-1,
        )

        idxs = ops.broadcast_to(
            ops.unsqueeze(
                dspark_kv_idxs(self.window, block_size, start_pos, x.device),
                0,
            ),
            [
                x.shape[0],
                block_size,
                min(self.window, start_pos + 1) + block_size,
            ],
        )
        o = sparse_attention(
            q,
            ops.concat([kv_cache, kv], axis=1),
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
