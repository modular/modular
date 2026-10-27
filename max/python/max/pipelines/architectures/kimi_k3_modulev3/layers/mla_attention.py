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
"""Kimi K3's multi-latent attention (ModuleV3).

``forward`` is the padded/parity form K3's own 0.4B fixture validates
against. ``forward_ragged`` and ``forward_latent_ragged`` serve over a real
paged KV cache instead -- the former decompressed, the latter the
compressed-latent form K3 actually serves -- direct V3 ports of
``kimi_k3/layers/mla_attention.py``'s ``kimi_k3_mla_attention``,
``kimi_k3_mla_attention_ragged``, and ``kimi_k3_mla_attention_latent_ragged``.

Neither ``deepseekV3_modulev3``'s ``QuantizedLatentAttentionWithRope`` nor the
generic V3 building block (``max.experimental.nn.common_layers.multi_latent_attention.LatentAttentionWithRope``)
can be wrapped for this layer:

* NO POSITIONAL ENCODING ANYWHERE. ``mla_use_nope`` is true: there is no
  ``apply_rotary_pos_emb`` call at all, and the ``qk_rope_head_dim``-wide
  slice is still split off and re-concatenated, UNROTATED.
* AN OUTPUT GATE neither V3 class has: ``attn *= sigmoid(g_proj(x))`` BEFORE
  ``o_proj``, gating on the layer's INPUT, not the attention output. Both V3
  classes call their own ``o_proj`` internally with no seam to insert this.

This block instead calls the same shared kernel-level primitive K3's own V2
parity form and the V3 attention building blocks both call
(``flash_attention_gpu``) directly, writing K3-native orchestration around it
-- the same pattern ``moe_latent.py`` uses for the MoE expert-compute side.

Two more facts this mirrors exactly from K3's V2 code and the official
reference, easy to get wrong silently:

* ``scaling`` is ``q_head_dim ** -0.5``, not ``v_head_dim ** -0.5``.
* ``k_rot`` (the rope-shaped slice) is MQA-shared -- one copy broadcast across
  every head -- while ``k_pass``/``v`` are per-head.

``q_head_dim`` and ``v_head_dim`` differ, so ``v`` is zero-padded to
``q_head_dim`` for the fused kernel and the output sliced back -- the
reference's own recipe, not a workaround.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flare_mla_prefill_plan,
    flash_attention_gpu,
    flash_attention_ragged,
    mla_prefill_decode_graph,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.experimental.nn.norm import RMSNorm
from max.experimental.sharding import DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams

from .tensor_parallel import check_divisible

# K3's q_a_layernorm/kv_a_layernorm always use this epsilon, independent of
# the model's general `rms_norm_eps` -- see `kimi_k3/layers/norm.py`.
_KIMI_QKV_A_LAYERNORM_EPS = 1e-6


class KimiK3MLAAttention(Module[..., Tensor]):
    """One ``KimiMLAAttention`` layer: no rotation, output-gated.

    ``forward`` is padded/parity. ``forward_ragged`` and
    ``forward_latent_ragged`` serve the same layer over a real paged KV
    cache, decompressed and compressed-latent respectively -- see the module
    docstring.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        kv_params: KVCacheParams | None = None,
        layer_idx: int = 0,
    ) -> None:
        """Args:
        hidden_size: ``config.hidden_size``.
        num_heads: ``config.num_attention_heads``.
        qk_nope_head_dim: ``config.qk_nope_head_dim`` -- the un-encoded
            query/key width per head.
        qk_rope_head_dim: ``config.qk_rope_head_dim`` -- the width of the
            MQA-shared slice a RoPE-shaped port would rotate; this model
            never does.
        v_head_dim: ``config.v_head_dim``.
        q_lora_rank: ``config.q_lora_rank``.
        kv_lora_rank: ``config.kv_lora_rank`` -- latent width of the
            compressed key/value.
        kv_params: Required for :meth:`forward_ragged` and
            :meth:`forward_latent_ragged`; this layer's leaf of the model's
            cache tree.
        layer_idx: This layer's position among the model's full-attention
            layers, for :meth:`forward_ragged` and
            :meth:`forward_latent_ragged`.
        """
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.kv_params = kv_params
        self.layer_idx = layer_idx

        self.q_a_proj = Linear(
            in_dim=hidden_size, out_dim=q_lora_rank, bias=False
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=_KIMI_QKV_A_LAYERNORM_EPS)
        self.q_b_proj = Linear(
            in_dim=q_lora_rank,
            out_dim=num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = Linear(
            in_dim=hidden_size,
            out_dim=kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank, eps=_KIMI_QKV_A_LAYERNORM_EPS
        )
        self.kv_b_proj = Linear(
            in_dim=kv_lora_rank,
            out_dim=num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.g_proj = Linear(
            in_dim=hidden_size, out_dim=num_heads * v_head_dim, bias=False
        )
        self.o_proj = Linear(
            in_dim=num_heads * v_head_dim, out_dim=hidden_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``[batch, seq_len, hidden]`` layer input -- also the source
                of the output gate, so it is read twice.

        Returns:
            ``[batch, seq_len, hidden]``.
        """
        batch, seq_len = x.shape[0], x.shape[1]

        q_lat = self.q_a_layernorm(self.q_a_proj(x))
        q = F.reshape(
            self.q_b_proj(q_lat),
            [batch, seq_len, self.num_heads, self.q_head_dim],
        )
        # This is the identity, since nothing is applied to the rope slice.
        # Writing it out (instead of just `query = q`) keeps this the exact
        # site where a RoPE implementation would rotate the slice.
        query = F.concat(
            [
                q[..., : self.qk_nope_head_dim],
                q[..., self.qk_nope_head_dim :],
            ],
            axis=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv = F.reshape(
            self.kv_b_proj(
                self.kv_a_layernorm(compressed_kv[..., : self.kv_lora_rank])
            ),
            [
                batch,
                seq_len,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ],
        )
        if kv.is_distributed:
            # The slices below need a resolved placement.
            kv = kv.rebind_mapping(
                NamedMapping(kv.mesh, (None, None, TP, None))
            )
        k_rot = F.broadcast_to(
            F.unsqueeze(compressed_kv[..., self.kv_lora_rank :], 2),
            [batch, seq_len, self.num_heads, self.qk_rope_head_dim],
        )
        if k_rot.is_distributed:
            # `concat` does not reshard its replicated operand to match.
            k_rot = k_rot.to(NamedMapping(k_rot.mesh, (None, None, TP, None)))
        key = F.concat([kv[..., : self.qk_nope_head_dim], k_rot], axis=-1)
        value = kv[..., self.qk_nope_head_dim :]

        attn = flash_attention_gpu(
            query,
            key,
            F.pad(
                value,
                [0, 0, 0, 0, 0, 0, 0, self.q_head_dim - self.v_head_dim],
            ),
            MHAMaskVariant.CAUSAL_MASK,
            float(self.q_head_dim**-0.5),
        )
        # `flash_attention_gpu` has no sharding rule, so restate the
        # placement before slicing.
        attn = attn.rebind_mapping(
            NamedMapping(attn.mesh, (None, None, TP, None))
        )
        attn = F.reshape(
            attn[..., : self.v_head_dim],
            [batch, seq_len, self.num_heads * self.v_head_dim],
        )

        gate = F.sigmoid(self.g_proj(x))
        return self.o_proj(attn * gate)

    def forward_ragged(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Serves this layer over a decompressed (per-head K/V) paged cache.

        Same arithmetic as :meth:`forward`, ragged instead of padded, with
        K and V materialized per head and written to the cache so a decode
        step attends over history instead of recomputing it. Costs
        ``2 * num_heads * q_head_dim`` bytes per token per layer against
        :meth:`forward_latent_ragged`'s ``kv_lora_rank + qk_rope_head_dim``
        -- use this only where the latent path's kernel constraints bar it;
        see ``model_config.py``'s ``supports_mla_latent_cache``.

        Args:
            x: ``[total_seq_len, hidden]`` ragged layer input.
            kv_collection: Paged cache handle for this batch.
            input_row_offsets: ``[batch_size + 1]`` exclusive prefix sums.

        Returns:
            ``[total_seq_len, hidden]``.
        """
        assert self.kv_params is not None, (
            "KimiK3MLAAttention needs kv_params to serve forward_ragged"
        )
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        q_lat = self.q_a_layernorm(self.q_a_proj(x))
        query = F.reshape(
            self.q_b_proj(q_lat),
            [total_seq_len, self.num_heads, self.q_head_dim],
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv = F.reshape(
            self.kv_b_proj(
                self.kv_a_layernorm(compressed_kv[:, : self.kv_lora_rank])
            ),
            [
                total_seq_len,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ],
        )
        if kv.is_distributed:
            # The slices below need a resolved placement.
            kv = kv.rebind_mapping(NamedMapping(kv.mesh, (None, TP, None)))
        k_rot = F.broadcast_to(
            F.unsqueeze(compressed_kv[:, self.kv_lora_rank :], 1),
            [total_seq_len, self.num_heads, self.qk_rope_head_dim],
        )
        if k_rot.is_distributed:
            # `concat` does not reshard its replicated operand to match.
            k_rot = k_rot.to(NamedMapping(k_rot.mesh, (None, TP, None)))
        key = F.concat([kv[:, :, : self.qk_nope_head_dim], k_rot], axis=-1)
        value = F.pad(
            kv[:, :, self.qk_nope_head_dim :],
            [0, 0, 0, 0, 0, self.q_head_dim - self.v_head_dim],
        )

        store_k_cache_ragged(kv_collection, key, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, value, input_row_offsets, layer_idx)

        attn = flash_attention_ragged(
            self.kv_params,
            input=query,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=float(self.q_head_dim**-0.5),
        )
        if attn.is_distributed:
            attn = attn.rebind_mapping(
                NamedMapping(attn.mesh, (None, TP, None))
            )
        attn = F.reshape(
            attn[:, :, : self.v_head_dim],
            [total_seq_len, self.num_heads * self.v_head_dim],
        )

        gate = F.sigmoid(self.g_proj(x))
        return self.o_proj(attn * gate)

    def forward_latent_ragged(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        prefill_plan: MLAPrefillMetadata,
        freqs_cis: Tensor,
    ) -> Tensor:
        """K3's production serving form: ragged, over a compressed latent cache.

        Same arithmetic as :meth:`forward`, reached a different way. The cache
        holds the COMPRESSED latent -- one ``kv_lora_rank + qk_rope_head_dim``
        vector per token, not per-head K and V, 64x fewer bytes -- so
        ``kv_b_proj`` is absorbed into the kernel instead of materialized:
        prefill decompresses a bounded window of the cache through ``w_k`` and
        ``w_uv``; decode projects query into the latent space through ``w_uk``
        and projects the attention output back out through ``w_uv``, so no
        per-head K or V is ever formed.

        Args:
            x: ``[total_seq_len, hidden]`` ragged layer input.
            kv_collection: Paged cache handle for this batch, from an MLA
                cache (``n_kv_heads`` 1, ``head_dim`` ``kv_lora_rank +
                qk_rope_head_dim``).
            input_row_offsets: ``[batch_size + 1]`` exclusive prefix sums.
            prefill_plan: This device's one-shot prefill plan, shared across
                every MLA layer this step -- see
                :func:`kimi_k3_mla_prefill_plan`.
            freqs_cis: The identity rotation -- see
                :func:`kimi_k3_nope_freqs_cis`.

        Returns:
            ``[total_seq_len, hidden]``.
        """
        assert self.kv_params is not None, (
            "KimiK3MLAAttention needs kv_params to serve forward_latent_ragged"
        )
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        q_lat = self.q_a_layernorm(self.q_a_proj(x))
        query = F.reshape(
            self.q_b_proj(q_lat),
            [total_seq_len, self.num_heads, self.q_head_dim],
        )

        # UNNORMALIZED: the kernel normalizes the latent half in place in the
        # cache after storing it, so normalizing here would apply the gain
        # twice.
        kv = self.kv_a_proj_with_mqa(x)

        # `kv_b_proj.weight` is `[heads * (qk_nope + v_head_dim), kv_lora_rank]`
        # with each head's rows contiguous, so a row-sharded TP copy keeps
        # whole heads and this reshape holds per device.
        kv_b = F.reshape(
            self.kv_b_proj.weight.T,
            [
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ],
        )
        if kv_b.is_distributed:
            kv_b = kv_b.rebind_mapping(
                NamedMapping(kv_b.mesh, (None, TP, None))
            )
        w_uk_base = kv_b[:, :, : self.qk_nope_head_dim]

        attn = mla_prefill_decode_graph(
            q=query,
            kv=kv,
            input_row_offsets=input_row_offsets,
            freqs_cis=F.cast(freqs_cis, query.dtype),
            kv_norm_gamma=self.kv_a_layernorm.weight,
            buffer_row_offsets=prefill_plan.buffer_row_offsets,
            cache_offsets=prefill_plan.cache_offsets,
            buffer_length=prefill_plan.buffer_lengths,
            # The same K up-projection twice: `[heads * qk_nope, rank]` for
            # prefill's plain matmul, `[heads, qk_nope, rank]` for decode's
            # per-head batched one.
            w_k=F.reshape(
                F.permute(w_uk_base, [1, 2, 0]), [-1, self.kv_lora_rank]
            ),
            w_uk=F.transpose(w_uk_base, 0, 1),
            w_uv=F.permute(kv_b[:, :, self.qk_nope_head_dim :], [1, 2, 0]),
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=float(self.q_head_dim**-0.5),
            epsilon=_KIMI_QKV_A_LAYERNORM_EPS,
            v_head_dim=self.v_head_dim,
            scalar_args=_require_mla_cache_field(
                kv_collection.attention_dispatch_metadata,
                "attention_dispatch_metadata",
            ),
            num_partitions_scalar=_require_mla_cache_field(
                kv_collection.mla_num_partitions, "mla_num_partitions"
            ),
        )
        if attn.is_distributed:
            attn = attn.rebind_mapping(
                NamedMapping(attn.mesh, (None, TP, None))
            )
        attn = F.reshape(
            attn, [total_seq_len, self.num_heads * self.v_head_dim]
        )

        gate = F.sigmoid(self.g_proj(x))
        return self.o_proj(attn * gate)


def kimi_k3_mla_prefill_plan(
    *,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    input_row_offsets: Tensor,
    buffer_size: int,
) -> MLAPrefillMetadata:
    """Plans the one-shot decompression buffer the latent prefill kernel fills.

    Reads only cache lengths and row offsets, neither of which varies by
    layer, so call this once per device per step and share the result across
    every :meth:`KimiK3MLAAttention.forward_latent_ragged` call that step --
    not once per layer.
    """
    layer_idx = F.constant(0, DType.uint32, device=CPU())
    buffer_row_offsets, cache_offsets, buffer_lengths = flare_mla_prefill_plan(
        kv_params,
        input_row_offsets,
        kv_collection,
        layer_idx,
        buffer_size,
        max_chunks=1,
    )
    # The kernel reads this scalar on the host; the plan op produces it on
    # device.
    mesh = buffer_lengths.mesh
    buffer_lengths = buffer_lengths.to(
        DeviceMesh(
            (CPU(),) * mesh.num_devices, mesh.mesh_shape, mesh.axis_names
        )
    )
    return MLAPrefillMetadata(
        buffer_row_offsets=buffer_row_offsets,
        cache_offsets=cache_offsets,
        buffer_lengths=buffer_lengths,
    )


def kimi_k3_nope_freqs_cis(
    *, max_seq_len: int, qk_rope_head_dim: int, dtype: DType, device: Device
) -> Tensor:
    """Returns the identity rotation the fused MLA kernels apply as their RoPE.

    K3 is NoPE: the rope-shaped slice is split off and re-concatenated
    unrotated, but the fused MLA kernels have no no-rope path -- they always
    rotate q and the key cache by ``freqs_cis``, interleaved as (cos, sin)
    pairs. Feeding cos=1, sin=0 makes that rotation exactly the identity,
    bit-exact in every float format -- the same trick
    ``fused_qk_rope.get_identity_rope_coeff`` uses in-kernel.
    """
    identity = np.zeros((max_seq_len, qk_rope_head_dim), dtype=np.float32)
    identity[:, 0::2] = 1.0
    return Tensor(identity, device=device).cast(dtype)


def _require_mla_cache_field(value: Tensor | None, name: str) -> Tensor:
    """Names the missing MLA cache input rather than tripping an assert later."""
    if value is None:
        raise ValueError(
            f"Kimi K3's latent MLA layers need {name}, which only an MLA KV"
            " cache provides."
        )
    return value


def tensor_parallel_mla_attention(
    layer: KimiK3MLAAttention, mesh: DeviceMesh
) -> KimiK3MLAAttention:
    """Shards an MLA layer by head along the ``tp`` mesh axis.

    Follows ``deepseekV3_modulev3``'s
    ``tensor_parallel_latent_attention_with_rope``. The latent projections
    ``q_a_proj`` and ``kv_a_proj_with_mqa`` stay replicated. ``q_b_proj``,
    ``kv_b_proj`` and the output gate ``g_proj`` are sharded on their output
    dim. ``o_proj`` is sharded on its input dim, so the caller must
    allreduce the output.
    """
    tp_size = mesh.axis_size(TP)
    if tp_size == 1:
        return layer
    check_divisible("MLA head count", layer.num_heads, tp_size)

    row_mapping = NamedMapping(mesh, (TP, None))
    layer.q_b_proj.weight = layer.q_b_proj.weight.to(row_mapping)
    layer.kv_b_proj.weight = layer.kv_b_proj.weight.to(row_mapping)
    layer.g_proj.weight = layer.g_proj.weight.to(row_mapping)
    layer.o_proj.weight = layer.o_proj.weight.to(NamedMapping(mesh, (None, TP)))
    return layer
