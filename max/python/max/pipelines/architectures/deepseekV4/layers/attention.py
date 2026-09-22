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

"""DeepSeek-V4 attention: shared-latent MQA over a sliding window plus CSA.

Reference: ``inference/model.py`` class ``Attention``. This file covers the
prefill path (``start_pos == 0``), which reads from freshly computed tensors
rather than the KV cache, so it does not depend on how compressed entries are
stored. Decode lands with the cache work.

Shape story, since it differs from V3.2 MLA at every step:

* ``wq_a`` -> ``q_norm`` -> ``wq_b`` gives ``[b, s, heads, head_dim]``. Then a
  *weightless* RMS normalize over ``head_dim`` -- ``q *= rsqrt(q.square().mean(-1)
  + eps)`` in the reference, with no learned gain, so it is not an ``RMSNorm``.
* ``wkv`` gives one ``[b, s, head_dim]`` latent shared by every head, serving as
  both key and value. There is no ``kv_b_proj``.
* RoPE covers the trailing ``qk_rope_head_dim`` dims of both.
* The non-RoPE dims of the latent go through a fused FP8 quantize/dequantize.
  This is not an optimization -- the model was trained with it (QAT), and vLLM
  refuses to run V4 with anything but an FP8 KV cache. Dropping it changes the
  numerics.
* The attention *output*'s RoPE dims are rotated by the conjugate before the
  output projection.
* The output projection is grouped and low-rank: reshape to ``o_groups``, einsum
  against ``wo_a`` viewed ``[groups, o_lora_rank, -1]``, flatten, then ``wo_b``.

Which compressed entries a query may see depends on the layer's ratio. Ratio-128
layers take every entry that closed before them. Ratio-4 layers ask the lightning
indexer, which scores the entries and keeps the best ``index_topk``.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from ..model_config import DeepseekV4Config
from .compressor import DeepseekV4Compressor
from .indexer import DeepseekV4Indexer, compress_cutoff
from .quantization import fp8_qat_quantize
from .rope import apply_rope_tail, rope_for_layer
from .sparse_attention import sparse_attention


def weightless_rms_normalize(x: TensorValue, eps: float) -> TensorValue:
    """``x * rsqrt(mean(x^2) + eps)`` over the last axis, with no learned gain.

    The reference writes this inline as
    ``q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)`` right
    after reshaping to heads. It is not an ``RMSNorm``: there is no weight.
    """
    x32 = ops.cast(x, DType.float32)
    scale = ops.rsqrt(ops.mean(x32 * x32, axis=-1) + eps)
    return ops.cast(x32 * scale, x.dtype)


def window_topk_idxs(
    seq_len: int, window: int, device: DeviceRef
) -> TensorValue:
    """Prefill sliding-window indices, ``[seq_len, min(seq_len, window)]``.

    ``get_window_topk_idxs(..., start_pos=0)``::

        base = arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + arange(min(seqlen, window))
        matrix = where(matrix > base, -1, matrix)

    So query ``i`` sees ``max(0, i-window+1) .. i``, padded with ``-1``.
    """
    span = min(seq_len, window)
    base = ops.unsqueeze(
        ops.range(
            0, seq_len, 1, out_dim=seq_len, device=device, dtype=DType.int32
        ),
        1,
    )
    offsets = ops.range(
        0, span, 1, out_dim=span, device=device, dtype=DType.int32
    )
    zero = ops.constant(0, DType.int32, device)
    start = ops.max(base - (window - 1), zero)
    matrix = start + offsets
    return ops.where(
        matrix > base, ops.constant(-1, DType.int32, device), matrix
    )


def compress_topk_idxs(
    seq_len: int, ratio: int, offset: int, device: DeviceRef
) -> TensorValue:
    """Prefill compressed-entry indices, ``[seq_len, seq_len // ratio]``.

    ``get_compress_topk_idxs(..., start_pos=0)``::

        matrix = arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = where(mask, -1, matrix + offset)

    Query ``i`` sees compressed entries ``j < (i + 1) // ratio`` -- every window
    that closed strictly before it. Used on ``compress_ratio == 128`` layers,
    which have no indexer and take every available entry.
    """
    n_compressed = seq_len // ratio
    cols = ops.range(
        0,
        n_compressed,
        1,
        out_dim=n_compressed,
        device=device,
        dtype=DType.int32,
    )
    cutoff = compress_cutoff(seq_len, ratio, device)
    return ops.where(
        cols >= cutoff,
        ops.constant(-1, DType.int32, device),
        cols + offset,
    )


class DeepseekV4Attention(Module):
    """Shared-latent MQA with sliding window and optional CSA compression."""

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        device: DeviceRef,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.compress_ratio = config.layer_compress_ratio(layer_idx)
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window = config.sliding_window
        self.eps = config.rms_norm_eps
        self.softmax_scale = config.head_dim**-0.5

        self.attn_sink = Weight(
            name="attn_sink",
            dtype=DType.float32,
            shape=(self.n_heads,),
            device=device,
        )
        self.wq_a = Linear(
            config.hidden_size, config.q_lora_rank, config.dtype, device
        )
        self.q_norm = RMSNorm(
            config.q_lora_rank, config.dtype, config.rms_norm_eps
        )
        self.wq_b = Linear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            config.dtype,
            device,
        )
        self.wkv = Linear(
            config.hidden_size, self.head_dim, config.dtype, device
        )
        self.kv_norm = RMSNorm(self.head_dim, config.dtype, config.rms_norm_eps)
        self.wo_a = Linear(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            config.dtype,
            device,
        )
        self.wo_b = Linear(
            self.o_groups * self.o_lora_rank,
            config.hidden_size,
            config.dtype,
            device,
        )

        self.rope = rope_for_layer(config, layer_idx, max_seq_len)
        # Graph-build-time stash, written by ``prefill`` and read by the model
        # right after the layer runs. Decode needs the prefill-final cache
        # state, and the graph is functional, so these intermediates have to
        # leave through ``graph.output``; stashing keeps every ``__call__``
        # signature unchanged for the committed gates. Dead when the caller
        # does not output them.
        self.exported: dict[str, TensorValue] = {}
        self.compressor = (
            DeepseekV4Compressor(
                config, self.compress_ratio, self.head_dim, device
            )
            if self.compress_ratio
            else None
        )
        # Only ratio-4 layers have an indexer. Ratio-128 layers select their
        # compressed entries by the strided rule instead.
        self.indexer = (
            DeepseekV4Indexer(config, self.compress_ratio, device)
            if self.compress_ratio == 4
            else None
        )

    def _output_projection(self, o: TensorValue) -> TensorValue:
        """``[b, s, heads, head_dim]`` -> ``[b, s, hidden_size]``.

        ``o.view(b, s, n_groups, -1)`` then
        ``einsum("bsgd,grd->bsgr", o, wo_a.view(n_groups, o_lora_rank, -1))``
        then ``wo_b(o.flatten(2))``. ``wo_a`` is used as a raw weight here, not
        called as a Linear.
        """
        b, s = o.shape[0], o.shape[1]
        group_width = self.n_heads * self.head_dim // self.o_groups
        # The group axis is the only batch dim, tokens are the rows: a rank-3
        # [g, b*s, d] @ [g, d, r] with no broadcast on either side. The
        # earlier [b*s, g, 1, d] @ [1, g, d, r] form broadcast wo_a over every
        # token, and the graph compiler materializes that broadcast as a
        # constant per layer -- 24.75 GB each at seq=198 -- which ran the B200
        # out of memory at model setup (ISSUES.md Issue 32).
        grouped = ops.transpose(
            ops.reshape(o, [b * s, self.o_groups, group_width]), 0, 1
        )
        wo_a = ops.reshape(
            self.wo_a.weight,
            [self.o_groups, self.o_lora_rank, group_width],
        )
        # [g, b*s, d] @ [g, d, r] -> [g, b*s, r]
        projected = ops.matmul(grouped, ops.transpose(wo_a, -1, -2))
        flattened = ops.reshape(
            ops.transpose(projected, 0, 1),
            [b, s, self.o_groups * self.o_lora_rank],
        )
        return self.wo_b(flattened)

    def prefill(self, x: TensorValue, seq_len: int) -> TensorValue:
        """Attention over a whole sequence starting at position 0.

        Args:
            x: ``[batch, seq_len, hidden_size]``, already ``attn_norm``-ed.
            seq_len: Static sequence length.
        """
        device = x.device
        rd = self.rope_head_dim
        freqs_cis = self.rope.freqs_cis_base()[:seq_len]

        # Query: low-rank, normed, per-head normed again (weightless), rotated.
        qr = self.q_norm(self.wq_a(x))
        q = ops.reshape(
            self.wq_b(qr),
            [x.shape[0], seq_len, self.n_heads, self.head_dim],
        )
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)

        # The shared latent: one row per token, key and value both.
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rope_tail(kv, freqs_cis, rd)
        # Positive bounds: see the note in rope.py and ISSUES Issue 34. This
        # exact expression, with ``:-rd``, is what produced a latent rolled by
        # rd and a step-0 cosine distance of 1.2e-02.
        kv_width = int(kv.shape[-1])
        kv = ops.concat(
            [
                fp8_qat_quantize(kv[..., : kv_width - rd]),
                kv[..., kv_width - rd :],
            ],
            axis=-1,
        )

        self.exported = {"latent": kv}
        if self.compressor is not None:
            comp_kv, comp_score = self.compressor.projections(
                ops.cast(x, DType.float32)
            )
            self.exported["comp_kv"] = comp_kv
            self.exported["comp_score"] = comp_score

        idxs = ops.broadcast_to(
            ops.unsqueeze(window_topk_idxs(seq_len, self.window, device), 0),
            [x.shape[0], seq_len, min(seq_len, self.window)],
        )
        # A sequence shorter than the ratio closes no window, so there is
        # nothing to select and nothing to append. The reference reaches the
        # same place by arithmetic rather than by a branch: Compressor.forward
        # returns None, and both index builders produce a width-zero matrix
        # that concatenates to nothing. Spelled out here because the graph
        # cannot carry a zero-width dimension through a gather.
        if self.compressor is not None and seq_len >= self.compress_ratio:
            compressed = self.compressor(x, seq_len, freqs_cis)
            self.exported["zone"] = compressed
            # Compressed entries are appended to the latent, so their indices
            # start where the per-token rows end -- hence the ``seq_len``
            # offset on both selection paths.
            if self.indexer is not None:
                compress_idxs = self.indexer(x, qr, seq_len, freqs_cis, seq_len)
            else:
                compress_idxs = ops.broadcast_to(
                    ops.unsqueeze(
                        compress_topk_idxs(
                            seq_len, self.compress_ratio, seq_len, device
                        ),
                        0,
                    ),
                    [x.shape[0], seq_len, seq_len // self.compress_ratio],
                )
            idxs = ops.concat([idxs, compress_idxs], axis=-1)
            kv = ops.concat([kv, compressed], axis=1)

        o = sparse_attention(q, kv, self.attn_sink, idxs, self.softmax_scale)
        o = apply_rope_tail(o, freqs_cis, rd, inverse=True)
        return self._output_projection(o)

    def decode_token(
        self,
        x: TensorValue,
        pos: TensorValue,
        ring: TensorValue,
        ring_pos: TensorValue,
        win_idxs: TensorValue,
        zone: TensorValue | None = None,
        comp_idxs: TensorValue | None = None,
        kv_state: TensorValue | None = None,
        score_state: TensorValue | None = None,
        ape_idx: TensorValue | None = None,
        should: TensorValue | None = None,
        zone_pos: TensorValue | None = None,
        comp_pos: TensorValue | None = None,
    ) -> tuple[
        TensorValue,
        TensorValue,
        TensorValue | None,
        TensorValue | None,
        TensorValue | None,
    ]:
        """Attention over one decode token, reference ``start_pos > 0`` branch.

        The caller holds every buffer and passes it in; the graph returns the
        updated ones (ISSUES Issue 30 / DECISIONS D17: MAX's paged cache cannot
        express the compressed zone, so the 9b numerical gate runs on
        caller-held state entirely).

        Position-derived scalars are host-computed inputs, mirroring the
        reference's ``lru_cache`` index builders, so one compiled graph serves
        every prompt and step:

        Args:
            x: ``[b, 1, hidden_size]``, already ``attn_norm``-ed.
            pos: ``[1]`` int32, the absolute position of this token
                (``start_pos`` in the reference).
            ring: ``[b, window, head_dim]`` sliding-window ring buffer of
                quantized latents, laid out so position ``p`` sits at slot
                ``p % window``.
            ring_pos: ``[1]`` int32, ``pos % window``.
            win_idxs: ``[b, 1, window]`` int32, ``get_window_topk_idxs``'s
                decode row (``-1`` padded below ``window - 1`` tokens).
            zone: ``[b, Z, head_dim]`` compressed entries, positions
                ``[window:]`` of the reference's flat cache. ``Z`` is a fixed
                cap; slots at and past ``(pos + 1) // ratio`` are dead and the
                ``-1``s in ``comp_idxs`` keep attention away from them.
            comp_idxs: ``[b, 1, Z]`` int32, ``arange((pos + 1) // ratio) +
                window`` padded to ``Z`` with ``-1``. For ratio-4 layers this
                is *also* the indexer's output verbatim: with every compressed
                count in this exercise far below ``index_topk`` (52 vs 512)
                the top-k keeps every closed entry, and attention is
                order-invariant, so the strided rule is numerically exact and
                the indexer's own cache need not be carried through decode.
                The 10-step driver asserts that bound.
            kv_state, score_state, ape_idx, should, comp_pos: The compressor's
                decode arguments; see ``DeepseekV4Compressor.decode``.
            zone_pos: ``[1]`` int32, ``pos // ratio`` -- the zone slot a closed
                window lands in.

        Returns:
            ``(out, new_ring, new_zone, new_kv_state, new_score_state)``; the
            last three are ``None`` on ratio-0 layers.
        """
        device = x.device
        rd = self.rope_head_dim
        freqs_row = ops.gather(self.rope.freqs_cis_base(), pos, axis=0)

        qr = self.q_norm(self.wq_a(x))
        q = ops.reshape(
            self.wq_b(qr), [x.shape[0], 1, self.n_heads, self.head_dim]
        )
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_row, rd)

        kv = self.kv_norm(self.wkv(x))
        kv = apply_rope_tail(kv, freqs_row, rd)
        # Positive bounds: ISSUES Issue 34, same as the prefill path.
        kv_width = int(kv.shape[-1])
        kv = ops.concat(
            [
                fp8_qat_quantize(kv[..., : kv_width - rd]),
                kv[..., kv_width - rd :],
            ],
            axis=-1,
        )

        slots = ops.range(
            0,
            self.window,
            1,
            out_dim=self.window,
            device=device,
            dtype=DType.int32,
        )
        new_ring = ops.where(
            ops.reshape(slots == ring_pos, [1, self.window, 1]), kv, ring
        )

        new_zone: TensorValue | None = None
        new_kv_state: TensorValue | None = None
        new_score_state: TensorValue | None = None
        if self.compressor is not None:
            assert zone is not None and comp_idxs is not None
            assert kv_state is not None and score_state is not None
            assert ape_idx is not None and should is not None
            assert zone_pos is not None and comp_pos is not None
            comp_freqs_row = ops.gather(
                self.rope.freqs_cis_base(), comp_pos, axis=0
            )
            entry, new_kv_state, new_score_state = self.compressor.decode(
                x, ape_idx, kv_state, score_state, should, comp_freqs_row
            )
            z = int(zone.shape[1])
            zslots = ops.range(
                0, z, 1, out_dim=z, device=device, dtype=DType.int32
            )
            write = ops.logical_and(
                ops.reshape(zslots == zone_pos, [1, z, 1]),
                ops.reshape(should, [1, 1, 1]),
            )
            new_zone = ops.where(write, entry, zone)
            kv_all = ops.concat([new_ring, new_zone], axis=1)
            idxs = ops.concat([win_idxs, comp_idxs], axis=-1)
        else:
            kv_all = new_ring
            idxs = win_idxs

        o = sparse_attention(
            q, kv_all, self.attn_sink, idxs, self.softmax_scale
        )
        o = apply_rope_tail(o, freqs_row, rd, inverse=True)
        return (
            self._output_projection(o),
            new_ring,
            new_zone,
            new_kv_state,
            new_score_state,
        )

    def __call__(self, x: TensorValue, seq_len: int) -> TensorValue:
        return self.prefill(x, seq_len)
