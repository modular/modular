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

Reference: ``inference/model.py`` class ``Attention``. The reference has two
branches -- ``start_pos == 0`` reads freshly computed tensors, ``start_pos >
0`` reads the caches those tensors were written into. Here there is one path:
a chunk of ``s`` tokens starting at ``P = cache_lengths`` reads what is in the
paged cache, appends what it computes, and writes it back (``layers/cache.py``,
``layers/csa.py``). ``P == 0`` is prefill, ``s == 1`` is decode, and anything
else is a chunked prefill or a prefix-cache continuation.

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

With a cache, the fused ``latent_sparse_attention_ragged`` kernel reads a
query's window straight out of the window leaf and its compressed entries out
of the zone leaf by entry index, once this chunk's latents and candidate
entries have been stored into them. Without a cache (a fresh sequence, the
reference for the cache gates) attention is order-invariant over its selected
rows, so they are laid out in position order in one table -- ``[window rows,
the chunk's own latents, the chunk's candidate entries]`` -- and the
reference's ring-slot / zone-offset index arithmetic becomes "row ``i + o`` of
the table, or ``-1``".
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.kernels import latent_sparse_attention_ragged
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from ..model_config import DeepseekV4Config
from .cache import (
    KEY,
    CacheLeaf,
    DeepseekV4Cache,
    arange,
    idiv,
    row_offsets,
    scalar,
)
from .compressor import DeepseekV4Compressor
from .csa import CompressedStream, compressed_stream
from .indexer import DeepseekV4Indexer
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


def chunk_positions(
    cache_lengths: TensorValue | None,
    batch: int,
    seq_len: int,
    device: DeviceRef,
) -> TensorValue:
    """``[batch, seq_len]`` int32 absolute positions of a chunk's tokens."""
    if cache_lengths is None:
        start = ops.broadcast_to(scalar(0, device), [batch])
    else:
        # The runtime input carries a symbolic batch dim; the graph is built
        # for a static one.
        start = ops.rebind(cache_lengths, [batch])
    return ops.reshape(start, [batch, 1]) + ops.reshape(
        arange(seq_len, device), [1, seq_len]
    )


def window_table(
    leaf: CacheLeaf | None,
    layer: int,
    fresh: TensorValue,
    positions: TensorValue,
    window: int,
) -> tuple[TensorValue, TensorValue]:
    """The sliding-window rows a chunk attends to, and each query's indices.

    Returns the table ``[b, window - 1 + s, head_dim]`` -- the ``window - 1``
    positions before the chunk read from ``leaf`` (garbage where they do not
    exist), then the chunk's own rows -- and ``[b, s, window]`` int32 indices
    into it: query ``i`` sees rows ``i .. i + window - 1``, i.e. positions
    ``P + i - window + 1 .. P + i``, with ``-1`` below position 0.
    """
    b, s = int(fresh.shape[0]), int(fresh.shape[1])
    head_dim = int(fresh.shape[2])
    device = fresh.device
    p = ops.reshape(positions[:, 0], [b, 1])
    back = ops.reshape(
        arange(window - 1, device) - (window - 1), [1, window - 1]
    )
    if leaf is not None:
        cached = leaf.gather(layer, KEY, ops.max(p + back, scalar(0, device)))
    else:
        cached = ops.broadcast_to(
            ops.constant(0.0, fresh.dtype, device), [b, window - 1, head_dim]
        )
    table = ops.concat([cached, fresh], axis=1)
    rows = ops.reshape(arange(s, device), [s, 1]) + ops.reshape(
        arange(window, device), [1, window]
    )
    attended = ops.reshape(p, [b, 1, 1]) + ops.unsqueeze(rows - (window - 1), 0)
    idxs = ops.where(
        attended >= scalar(0, device),
        ops.broadcast_to(ops.unsqueeze(rows, 0), [b, s, window]),
        scalar(-1, device),
    )
    return table, idxs


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
        self.max_seq_len = max_seq_len
        # Where this layer's rows live: the window leaf is indexed by trunk
        # layer (DSpark stages follow the trunk), the compressed leaves by the
        # layer's rank among layers of the same ratio.
        self.swa_layer = layer_idx
        self.zone_layer = (
            config.compressed_layer_index(layer_idx)
            if self.compress_ratio
            else -1
        )

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

    def latent(self, x: TensorValue, freqs_cis: TensorValue) -> TensorValue:
        """The shared KV latent for a chunk: normed, rotated, FP8-simulated."""
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rope_tail(kv, freqs_cis, rd)
        # Positive bounds: see the note in rope.py and ISSUES Issue 34. This
        # exact expression, with ``:-rd``, is what produced a latent rolled by
        # rd and a step-0 cosine distance of 1.2e-02.
        kv_width = int(kv.shape[-1])
        return ops.concat(
            [
                fp8_qat_quantize(kv[..., : kv_width - rd]),
                kv[..., kv_width - rd :],
            ],
            axis=-1,
        )

    def __call__(
        self,
        x: TensorValue,
        seq_len: int,
        cache: DeepseekV4Cache | None = None,
    ) -> TensorValue:
        """Attention for a chunk of ``seq_len`` tokens.

        Args:
            x: ``[batch, seq_len, hidden_size]``, already ``attn_norm``-ed.
            seq_len: Static chunk length.
            cache: The paged leaves; the chunk starts at
                ``cache.cache_lengths`` and is appended to them. ``None`` runs
                a fresh sequence from position 0 without storing anything.
        """
        b = int(x.shape[0])
        s = seq_len
        device = x.device
        rd = self.rope_head_dim
        positions = chunk_positions(
            cache.cache_lengths if cache is not None else None, b, s, device
        )
        freqs_cis = ops.gather(self.rope.freqs_cis_base(), positions, axis=0)

        # Query: low-rank, normed, per-head normed again (weightless), rotated.
        qr = self.q_norm(self.wq_a(x))
        q = ops.reshape(self.wq_b(qr), [b, s, self.n_heads, self.head_dim])
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)

        kv = self.latent(x, freqs_cis)
        if cache is not None:
            cache.swa.store(
                self.swa_layer,
                KEY,
                ops.reshape(kv, [b * s, self.head_dim]),
                row_offsets(b, s, device),
            )

        stream: CompressedStream | None = None
        candidates: TensorValue | None = None
        if self.compressor is not None:
            ratio = self.compress_ratio
            x32 = ops.cast(x, DType.float32)
            stream = compressed_stream(
                self.compressor,
                x32,
                positions,
                self.rope.freqs_cis_base(),
                self.max_seq_len,
                cache.state.get(ratio) if cache is not None else None,
                cache.comp.get(ratio) if cache is not None else None,
                self.zone_layer,
            )
            # Query ``i`` at position ``t`` sees entries below ``(t + 1) //
            # ratio``: every window that closed strictly before it.
            cutoff = idiv(positions + scalar(1, device), ratio)
            valid = stream.valid(cutoff)
            if self.indexer is not None:
                idx_stream = compressed_stream(
                    self.indexer.compressor,
                    x32,
                    positions,
                    self.rope.freqs_cis_base(),
                    self.max_seq_len,
                    cache.idx_state if cache is not None else None,
                    cache.idx_comp if cache is not None else None,
                    self.zone_layer,
                )
                candidates = self.indexer(
                    x, qr, freqs_cis, idx_stream.table, valid
                )
            else:
                n_cand = stream.cap + stream.n_new
                candidates = ops.where(
                    valid,
                    ops.reshape(arange(n_cand, device), [1, 1, n_cand]),
                    scalar(-1, device),
                )

        if cache is None:
            o = self._attend_table(q, kv, positions, stream, candidates)
        else:
            o = self._attend_leaves(q, cache, stream, candidates)
        o = apply_rope_tail(o, freqs_cis, rd, inverse=True)
        return self._output_projection(o)

    def _attend_leaves(
        self,
        q: TensorValue,
        cache: DeepseekV4Cache,
        stream: CompressedStream | None,
        candidates: TensorValue | None,
    ) -> TensorValue:
        """The fused kernel over the window leaf and this layer's zone leaf.

        Both leaves already hold the chunk's rows, so the kernel needs no
        fresh operands: the window is addressed by position from
        ``cache_lengths`` and the compressed entries by the numbers
        ``stream.entries`` resolves.
        """
        b, s, h, d = (int(v) for v in q.shape)
        rows = b * s
        device = q.device
        if stream is not None:
            assert candidates is not None
            comp_leaf = cache.comp[self.compress_ratio]
            entries = ops.reshape(
                stream.entries(candidates), [rows, int(candidates.shape[-1])]
            )
            layer_comp = self.zone_layer
        else:
            # A window-only layer has no zone leaf; the kernel still takes a
            # compressed operand, so it gets the window leaf and no entries.
            comp_leaf = cache.swa
            entries = ops.constant(
                np.zeros((rows, 0), np.int32), DType.int32, device
            )
            layer_comp = self.swa_layer
        o = latent_sparse_attention_ragged(
            ops.reshape(q, [rows, h, d]),
            row_offsets(b, s, device),
            entries,
            self.attn_sink,
            cache.swa.values,
            comp_leaf.values,
            ops.constant(self.swa_layer, DType.uint32, DeviceRef.CPU()),
            ops.constant(layer_comp, DType.uint32, DeviceRef.CPU()),
            scale=self.softmax_scale,
            window=self.window,
        )
        return ops.reshape(o, [b, s, h, d])

    def _attend_table(
        self,
        q: TensorValue,
        kv: TensorValue,
        positions: TensorValue,
        stream: CompressedStream | None,
        candidates: TensorValue | None,
    ) -> TensorValue:
        """Graph-side attention over a gathered table, for a cache-less chunk."""
        table, idxs = window_table(
            None, self.swa_layer, kv, positions, self.window
        )
        if stream is not None:
            assert candidates is not None
            offset = scalar(int(table.shape[1]), positions.device)
            table = ops.concat([table, stream.table], axis=1)
            comp_idxs = ops.where(
                candidates >= scalar(0, positions.device),
                candidates + offset,
                scalar(-1, positions.device),
            )
            idxs = ops.concat([idxs, comp_idxs], axis=-1)
        return sparse_attention(
            q, table, self.attn_sink, idxs, self.softmax_scale
        )
