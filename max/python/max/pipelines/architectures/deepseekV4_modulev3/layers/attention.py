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
"""DeepSeek-V4 attention on ModuleV3, read from the paged leaves.

Shared-latent MQA over a sliding window plus compressed sparse attention
(CSA). One path for prefill (``cache_lengths == 0``), chunked prefill / prefix
resume and decode (``s == 1``): a chunk at ``P = cache_lengths`` stores its
latents and candidate entries, then the fused kernel reads the window by
position and the compressed entries by number. The shape story and the
reference mapping are in the ModuleV2 implementation
(``deepseekV4/layers/attention.py`` at dfc8109fc0c).
"""

from __future__ import annotations

import os

import numpy as np
from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.mesh_axis import TP
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from ..model_config import DeepseekV4Config
from .cache import KEY, DeepseekV4Cache, LayerIndex
from .compressor import DeepseekV4Compressor
from .csa import arange, compressed_stream, int32
from .indexer import DeepseekV4Indexer
from .kernels import latent_sparse_attention_ragged
from .norm import DeepseekV4RMSNorm, weightless_rms_normalize
from .quantization import (
    dequantize_fp8_blocks,
    fp8_qat_quantize,
    linear_for,
    shard_linear,
)
from .ragged import RaggedRows
from .rope import apply_rope_tail, rope_for_layer

__all__ = ["DeepseekV4Attention"]

# The V2 layer's kernel A/B switches, read once at graph build.
# TODO(dsv4-attn-core): drop with the V2 ones.
_LATENT_ATTN_SPLIT_K = (
    os.environ.get("MAX_DSV4_LATENT_ATTN_SPLIT_K", "1") != "0"
)
_LATENT_ATTN_PORTABLE_P_BF16 = (
    os.environ.get("MAX_DSV4_LATENT_ATTN_PORTABLE_P_BF16", "0") == "1"
)


class DeepseekV4Attention(Module[..., Tensor]):
    """Shared-latent MQA with sliding window and optional CSA compression."""

    attn_sink: Tensor

    def __init__(
        self, config: DeepseekV4Config, layer_idx: int, max_seq_len: int
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.compress_ratio = config.layer_compress_ratio(layer_idx)
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.window = config.sliding_window
        self.eps = config.rms_norm_eps
        self.softmax_scale = config.head_dim**-0.5
        self.max_seq_len = max_seq_len
        # The window leaf is indexed by trunk layer, the compressed leaves by
        # the layer's rank among layers of its ratio.
        self.swa_layer = layer_idx
        self.zone_layer = (
            config.compressed_layer_index(layer_idx)
            if self.compress_ratio
            else None
        )

        self.attn_sink = Tensor.zeros([self.n_heads], dtype=DType.float32)
        # ``[wq_a; wkv]`` stacked by the adapter: one GEMM, not two.
        self.wqa_wkv = linear_for(
            config, config.hidden_size, config.q_lora_rank + self.head_dim
        )
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, self.eps)
        self.wq_b = linear_for(
            config, config.q_lora_rank, self.n_heads * self.head_dim
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, self.eps)
        self.wo_a = linear_for(
            config,
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
        )
        self.wo_b = linear_for(
            config, self.o_groups * self.o_lora_rank, config.hidden_size
        )
        self.rope = rope_for_layer(config, layer_idx, max_seq_len)
        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio
            else None
        )
        # Ratio-128 layers take every closed entry; only ratio 4 ranks them.
        self.indexer = (
            DeepseekV4Indexer(config, self.compress_ratio)
            if self.compress_ratio == 4
            else None
        )
        if self.attn_sink.mesh.num_devices > 1:
            self._shard_heads()

    def _shard_heads(self) -> None:
        """The V2 tensor-parallel split (``DeepseekV4Attention.shard_heads``).

        ``wq_b`` and ``attn_sink`` per head, ``wo_a`` per group (a group is a
        run of whole heads) and ``wo_b`` along its input, so each device's
        output is a partial sum. The latent side (``wqa_wkv``, the norms, the
        compressor) is shared by every head and stays whole, as do the
        single-head KV leaves.
        """
        mesh = self.attn_sink.mesh
        n = mesh.num_devices
        if self.n_heads % n or self.o_groups % n:
            raise ValueError(
                f"{self.n_heads} heads in {self.o_groups} groups do not split "
                f"across {n} devices"
            )
        self.attn_sink = self.attn_sink.to(NamedMapping(mesh, (TP,)))
        shard_linear(self.wq_b, 0)
        shard_linear(self.wo_a, 0)
        shard_linear(self.wo_b, 1)
        if self.indexer is not None:
            self.indexer.shard_heads()

    def layer_index(self) -> LayerIndex:
        """This layer's place in the leaves, as constants."""
        return LayerIndex.constant(self.swa_layer, self.zone_layer)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(wq_a(x), wkv(x))`` sliced out of the one stacked projection."""
        out = self.wqa_wkv(x)
        width = int(out.shape[-1])
        return out[..., : self.q_lora_rank], out[..., self.q_lora_rank : width]

    def latent(self, kv: Tensor, freqs_cis: Tensor) -> Tensor:
        """The KV latent from ``wkv(x)``: normed, rotated, FP8-simulated."""
        rd = self.rope_head_dim
        kv = apply_rope_tail(self.kv_norm(kv), freqs_cis, rd)
        # Positive bounds: a ``:-rd`` slice here rolled the latent.
        width = int(kv.shape[-1])
        return F.concat(
            [fp8_qat_quantize(kv[..., : width - rd]), kv[..., width - rd :]],
            axis=-1,
        )

    def _output_projection(self, o: Tensor) -> Tensor:
        """Projects ``[T, heads, head_dim]`` to ``[T, hidden_size]``.

        The result is a partial sum when the heads are split. The group axis
        is the only batch dim of a rank-3 ``[g, T, d] @ [g, d, r]``:
        broadcasting ``wo_a`` over tokens instead materializes a 24.75 GB
        constant per layer.
        """
        t = o.shape[0]
        group_width = self.n_heads * self.head_dim // self.o_groups
        grouped = o.reshape([t, self.o_groups, group_width]).transpose(0, 1)
        weight = self.wo_a.weight
        scale = getattr(self.wo_a, "weight_scale", None)
        if scale is not None:
            weight = dequantize_fp8_blocks(weight, scale, o.dtype)
        wo_a = weight.reshape([self.o_groups, self.o_lora_rank, group_width])
        projected = grouped @ wo_a.transpose(-1, -2)
        flattened = projected.transpose(0, 1).reshape(
            [t, self.o_groups * self.o_lora_rank]
        )
        return self.wo_b(flattened)

    def forward(
        self,
        x: Tensor,
        rows: RaggedRows,
        cache: DeepseekV4Cache,
        index: LayerIndex,
    ) -> Tensor:
        """Runs cached attention on ``[T, hidden]`` tokens to ``[T, hidden]``.

        The input is already ``attn_norm``-ed. Under tensor parallelism the
        output is partial over the heads' devices.

        ``index`` is where this layer lives in the leaves: operands, not
        constants, so a shared subgraph reads its caller's leaves.
        """
        t = x.shape[0]
        rd = self.rope_head_dim
        freqs_table = self.rope.freqs_cis_base(x.mesh)
        freqs_cis = F.gather(freqs_table, rows.positions, axis=0)

        # Query: low-rank, normed, per-head normed again (weightless), rotated.
        q_a, kv_a = self.project(x)
        qr = self.q_norm(q_a)
        q = self.wq_b(qr).reshape([t, self.n_heads, self.head_dim])
        q = weightless_rms_normalize(q, self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)

        kv = self.latent(kv_a, freqs_cis)
        cache.swa.store(index.swa, KEY, kv, rows.offsets)

        if self.compressor is not None:
            ratio = self.compress_ratio
            assert index.zone is not None
            x32 = x.cast(DType.float32)
            windows = rows.windows[ratio]
            stream = compressed_stream(
                self.compressor,
                x32,
                rows,
                windows,
                freqs_table,
                self.max_seq_len,
                cache.state[ratio],
                cache.comp[ratio],
                index,
            )
            # A query at ``p`` sees the entries below ``(p + 1) // ratio``.
            cutoff = (rows.positions + int32(1, rows.positions)) // ratio
            valid = stream.valid(cutoff)
            if self.indexer is not None:
                assert cache.idx_state is not None
                assert cache.idx_comp is not None
                idx_stream = compressed_stream(
                    self.indexer.compressor,
                    x32,
                    rows,
                    windows,
                    freqs_table,
                    self.max_seq_len,
                    cache.idx_state,
                    cache.idx_comp,
                    index,
                )
                candidates = self.indexer(
                    x, qr, freqs_cis, idx_stream, cutoff, valid
                )
            else:
                n_cand = stream.n_cand
                candidates = F.where(
                    valid,
                    arange(n_cand, valid).reshape([1, n_cand]),
                    int32(-1, valid),
                )
            comp_values = cache.comp[ratio].values
            entries = stream.entries(candidates)
            layer_comp = index.zone
        else:
            # A window-only layer still hands the kernel a compressed operand:
            # the window leaf, with one skipped entry per row. A zero-width
            # entry tensor has no device pointer for the kernel to read.
            comp_values = cache.swa.values
            entries = F.broadcast_to(
                F.constant(
                    np.full((1, 1), -1, np.int32),
                    DType.int32,
                    device=x.mesh,
                ),
                [t, 1],
            )
            layer_comp = index.swa

        o = latent_sparse_attention_ragged(
            q,
            rows.offsets,
            entries,
            self.attn_sink,
            cache.swa.values,
            comp_values,
            index.swa,
            layer_comp,
            scale=self.softmax_scale,
            window=self.window,
            split_k=_LATENT_ATTN_SPLIT_K,
            portable_p_bf16=_LATENT_ATTN_PORTABLE_P_BF16,
        )
        o = apply_rope_tail(o, freqs_cis, rd, inverse=True)
        return self._output_projection(o)
