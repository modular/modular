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
"""The Kimi-K3 DSpark drafter (ModuleV3), TP-sharded like its target.

A DSpark block drafter: five dense layers of MLA attention over a
block-diffusion schedule. Each step

1. projects the target's tapped hidden states through ``context_proj`` and
   ``context_norm`` into one context stream (:meth:`K3DSpark.project_context`);
2. writes that stream's latent into the drafter's own paged cache, one row
   per verified token (:meth:`K3DSpark.materialize`) -- accepted or not,
   since the cache length advances by the committed rows only and the next
   step's store overwrites the rest;
3. runs the block -- the committed token followed by ``block_size - 1`` mask
   slots, embedded through the drafter's own ``embed_tokens`` -- through the
   layers (:meth:`K3DSpark.forward_block`), which each store the block rows'
   latent and attend non-causally over the context plus the whole block;
4. scores every block slot through the target's LM head
   (:meth:`K3DSpark.logits`), and the proposer chains the slots through the
   Markov head, each position's bias seeded by the token drawn before it.

Attention shares the target's latent shape -- the same fused MLA graph
ops over the same compressed cache -- and differs in its knobs:

* yarn RoPE on the rope slice -- the target applies none -- through the
  fused store, whose real rope table replaces the target's identity one;
* no output gate (``mla_use_output_gate=false``);
* dual-source latent: the context rows' latent comes from the projected
  taps, the block rows' from the evolving block hidden states, both through
  the same ``kv_a_proj_with_mqa`` and both stored ``cache_length + row`` in;
* ``NULL_MASK`` -- every block row attends to its request's whole cache,
  itself included, the block-diffusion mask.

The KV leaf is the compressed latent: one ``kv_lora_rank +
qk_rope_head_dim`` row per token, a single KV head, ``kv_b_proj`` absorbed
into the kernel as ``w_k``/``w_uk``/``w_uv`` -- exactly the target's own
latent path (``KimiK3MLAAttention.forward_latent_ragged``), which the
drafter's context store and non-causal block call share with it. The
decompressed per-head form this drafter first shipped with is gone; a
checkpoint whose latent the fused MLA kernels cannot serve is refused at
config time (``dspark/model_config.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding, Linear, Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.multi_latent_attention import MLAPrefillMetadata
from max.nn.kernels import mla_prefill_decode_graph
from max.nn.kernels.mla import store_mla_latent_ragged
from max.nn.kv_cache import MLAKVCacheParams, spec_decode_cache_slack
from max.nn.norm import RMSNorm
from max.nn.sequential import ModuleList
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping, Sharded

from ..layers.mla_attention import (
    _require_mla_cache_field,
    kimi_k3_mla_prefill_plan,
)
from ..layers.tensor_parallel import check_divisible
from .config import K3DSparkDraftConfig, yarn_get_mscale

__all__ = [
    "K3DSpark",
    "K3DSparkMLAAttention",
    "k3_dspark_yarn_freqs_cis",
    "tensor_parallel_k3_dspark",
]


def _greedy(logits: Tensor, step: int) -> Tensor:
    del step
    return F.squeeze(F.argmax(logits, axis=-1), axis=-1)


def _yarn_correction_dim(
    num_rotations: float, dim: int, base: float, max_position: int
) -> float:
    """The YaRN ramp bound: the correction dim for a given rotation count."""
    return (
        dim
        * math.log(max_position / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def k3_dspark_yarn_freqs_cis(
    config: K3DSparkDraftConfig, positions: int
) -> np.ndarray:
    """The yarn RoPE table for the drafter's rope slice, un-mscaled.

    ``[positions, qk_rope_head_dim]`` float32, row ``p`` holding the
    per-pair ``(cos, sin)`` of ``p * inv_freq`` -- the layout the fused rope
    kernel's interleaved rotation reads. The table carries no mscale on
    the published checkpoint (its config declares 1.0 for both fields);
    the whole correction rides the attention scale instead.

    Args:
        config: The drafter's config, whose rope fields shape the table.
        positions: Row count; one past the highest position any row can
            hold, so the block's tail stays in bounds.
    """
    dim = config.qk_rope_head_dim
    extra = 1.0 / (
        config.rope_theta
        ** (np.arange(0, dim, 2, dtype=np.float64) / np.float64(dim))
    )
    inter = extra / config.rope_factor
    low = max(
        float(
            np.floor(
                _yarn_correction_dim(
                    config.rope_beta_fast,
                    dim,
                    config.rope_theta,
                    config.rope_original_max_position_embeddings,
                )
            )
        ),
        0.0,
    )
    high = min(
        float(
            np.ceil(
                _yarn_correction_dim(
                    config.rope_beta_slow,
                    dim,
                    config.rope_theta,
                    config.rope_original_max_position_embeddings,
                )
            )
        ),
        float(dim - 1),
    )
    if high <= low:
        high = low + 0.001
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0.0, 1.0
    )
    # Fast rotations keep their frequencies; slow ones interpolate by the
    # yarn factor.
    inv_freq = (inter * ramp + extra * (1.0 - ramp)).astype(np.float32)
    mscale = yarn_get_mscale(
        config.rope_factor, config.rope_mscale
    ) / yarn_get_mscale(config.rope_factor, config.rope_mscale_all_dim)
    t = np.arange(positions, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.stack(
        [np.cos(freqs) * mscale, np.sin(freqs) * mscale], axis=-1
    ).reshape(positions, dim)


class K3DSparkMLAAttention(Module[..., Tensor]):
    """Dual-source MLA attention: yarn rope, no gate, non-causal blocks.

    Q comes from the rows the caller runs (context rows never do); the
    rows' latent lands in the drafter's own paged cache at ``cache_length +
    row``, normed and roped in place by the kernel, and the block rows'
    attention reads it back through the absorbed ``kv_b_proj`` -- the
    target's latent pattern, never a per-head K or V.
    """

    def __init__(
        self,
        config: K3DSparkDraftConfig,
        kv_params: MLAKVCacheParams,
        layer_idx: int,
    ) -> None:
        super().__init__()
        if not isinstance(kv_params, MLAKVCacheParams):
            raise ValueError(
                "The K3 DSpark drafter serves through the fused MLA kernels"
                " and needs an MLA latent KV leaf; got"
                f" {type(kv_params).__name__}."
            )
        self.config = config
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.q_head_dim = config.q_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        hidden = config.hidden_size
        eps = config.rms_norm_eps

        self.q_a_proj = Linear(
            in_dim=hidden, out_dim=config.q_lora_rank, bias=False
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=eps)
        self.q_b_proj = Linear(
            in_dim=config.q_lora_rank,
            out_dim=self.num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = Linear(
            in_dim=hidden,
            out_dim=config.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=eps)
        self.kv_b_proj = Linear(
            in_dim=config.kv_lora_rank,
            out_dim=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = Linear(
            in_dim=self.num_heads * self.v_head_dim,
            out_dim=hidden,
            bias=False,
        )

    def _layer(self) -> Tensor:
        return F.constant(self.layer_idx, DType.uint32, device=CPU())

    def _latent(self, x: Tensor) -> Tensor:
        """The rows' raw latent, ``[rows, kv_lora_rank + qk_rope_head_dim]``.

        UNNORMALIZED: every kernel that consumes it applies
        ``kv_a_layernorm``'s gain to the latent half in the cache, after
        storing it, so normalizing here would apply the gain twice.
        """
        return self.kv_a_proj_with_mqa(x)

    def materialize(
        self,
        ctx: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        rope_table: Tensor,
    ) -> None:
        """Stores the context rows' latent into the cache, normed and roped.

        The context rows are projected target taps -- stored but never
        queried, since the block call's queries are the block rows -- so
        this is the store-only op, not the attention op's store half.
        """
        store_mla_latent_ragged(
            kv=self._latent(ctx),
            input_row_offsets=input_row_offsets,
            freqs_cis=rope_table,
            kv_norm_gamma=self.kv_a_layernorm.weight,
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            layer_idx=self._layer(),
            epsilon=self.kv_a_layernorm.eps,
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        rope_table: Tensor,
        prefill_plan: MLAPrefillMetadata,
    ) -> Tensor:
        """Runs the block rows: store, rope, attend over context plus block.

        The target's ``forward_latent_ragged`` with the drafter's knobs: the
        YaRN rope table and scale instead of the identity table and
        ``q_head_dim ** -0.5``, ``NULL_MASK`` instead of causal, and no
        output gate.
        """
        total = x.shape[0]
        q = F.reshape(
            self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))),
            [total, self.num_heads, self.q_head_dim],
        )
        kv_mapping = NamedMapping(x.mesh, (None, TP, None))
        q = q.rebind_mapping(kv_mapping)

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
            q=q,
            kv=self._latent(x),
            input_row_offsets=input_row_offsets,
            # Already at the rows' dtype: `_rope` cast it once for every
            # consumer of this step's table.
            freqs_cis=rope_table,
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
            layer_idx=self._layer(),
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.config.attention_scale,
            epsilon=self.kv_a_layernorm.eps,
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
            attn = attn.rebind_mapping(kv_mapping)
        attn = F.reshape(attn, [total, self.num_heads * self.v_head_dim])
        # Partial sums under TP; the layer reduces before its residual.
        return self.o_proj(attn)


class K3DSparkMLP(Module[..., Tensor]):
    """The SiLU-gated MLP of a drafter layer."""

    def __init__(self, config: K3DSparkDraftConfig) -> None:
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(in_dim=hidden, out_dim=inner, bias=False)
        self.up_proj = Linear(in_dim=hidden, out_dim=inner, bias=False)
        self.down_proj = Linear(in_dim=inner, out_dim=hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class K3DSparkLayer(Module[..., Tensor]):
    """One pre-norm drafter layer over the block's rows."""

    def __init__(
        self,
        config: K3DSparkDraftConfig,
        kv_params: MLAKVCacheParams,
        layer_idx: int,
    ) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = K3DSparkMLAAttention(config, kv_params, layer_idx)
        self.mlp = K3DSparkMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        rope_table: Tensor,
        prefill_plan: MLAPrefillMetadata,
    ) -> Tensor:
        tp_active = TP in h.mesh.axis_names and h.mesh.axis_size(TP) > 1
        attn = self.self_attn(
            self.input_layernorm(h),
            kv_collection,
            input_row_offsets,
            rope_table,
            prefill_plan,
        )
        if tp_active:
            attn = F.allreduce_sum(attn, mesh_axis=TP)
        h = h + attn.to(h.mapping)
        out = self.mlp(self.post_attention_layernorm(h))
        if tp_active:
            out = F.allreduce_sum(out, mesh_axis=TP)
        return h + out.to(h.mapping)


class K3DSparkMarkovHead(Module[..., Tensor]):
    """The low-rank bigram bias, under the checkpoint's own names."""

    def __init__(self, config: K3DSparkDraftConfig) -> None:
        super().__init__()
        self.markov_w1 = Embedding(config.vocab_size, dim=config.markov_rank)
        self.markov_w2 = Linear(
            in_dim=config.markov_rank,
            out_dim=config.vocab_size,
            bias=False,
        )

    def bias_float32(self, previous: Tensor) -> Tensor:
        """``markov_w2(markov_w1[previous])`` computed in float32 on rank 0.

        The weights are replicated, so the head runs once, on the first
        device's copy, beside the chain that reads it. Widening before the
        matmul keeps the bias from rounding to bf16, which would move every
        step's argmax near ties.
        """
        embed = self.markov_w1.weight.local_shards[0]
        return F.gather(embed, previous, axis=0).cast(DType.float32) @ (
            self.markov_w2.weight.local_shards[0].cast(DType.float32).T
        )


class K3DSpark(Module[..., Tensor]):
    """The DSpark drafter: context projection, layers, Markov head.

    Holds no head of its own: the caller scores the block through the
    target's LM head weight (``logits``), as the published checkpoint
    ships none.
    """

    has_own_embed_tokens = True
    """The drafter embeds the block through its own embedding, never the
    target's: the fused pairing shares the target's module onto a draft
    only when this stays unset."""

    def __init__(
        self,
        config: K3DSparkDraftConfig,
        kv_params: MLAKVCacheParams,
        *,
        max_seq_len: int,
        mla_buffer_size: int,
    ) -> None:
        """Builds the drafter on the current default device and dtype.

        Args:
            config: The parsed ``k3_dspark`` checkpoint config.
            kv_params: The drafter's own MLA latent cache leaf.
            max_seq_len: The longest sequence, which sizes the rope table
                (with the pool's speculative slack and the block's rows).
            mla_buffer_size: The cap the block call's one-shot prefill plan
                clamps its window to, shared by every layer of one forward
                -- sized with the drafter's own head count and block width
                (``dspark/model_config.py``), never the target's.
        """
        super().__init__()
        self.config = config
        self.kv_params = kv_params
        self.block_size = config.block_size
        self.mla_buffer_size = mla_buffer_size
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.context_proj = Linear(
            in_dim=config.num_context_features * hidden,
            out_dim=hidden,
            bias=False,
        )
        self.context_norm = RMSNorm(hidden, eps=eps)
        self.embed_tokens = Embedding(config.vocab_size, dim=hidden)
        self.layers = ModuleList(
            [
                K3DSparkLayer(config, kv_params, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden, eps=eps)
        self.markov_head = K3DSparkMarkovHead(config)
        # Derived, not loaded: held as a plain array so it rides the module
        # tree without becoming a parameter the checkpoint must name. A
        # context-boundary request may over-speculate past ``max_seq_len``
        # into the pool's whole ``spec_decode_cache_slack``, and the block
        # call then writes its own ``block_size`` rows beyond that, so the
        # table covers both and no row's position reads out of bounds.
        self._rope_table = k3_dspark_yarn_freqs_cis(
            config,
            max_seq_len
            + spec_decode_cache_slack(kv_params)
            + config.block_size,
        )

    def _rope(self, like: Tensor) -> Tensor:
        """The rope table beside the rows it rotates, at the rows' dtype.

        Cast once here so every consumer -- the context store, the block
        call's store, and the block rows' queries -- reads the same
        coefficients. The target does the same cast at its own call site
        (``F.cast(freqs_cis, query.dtype)``); an f32 table beside bf16 rows
        would round the two stores' rotations differently.
        """
        target = like.mesh if like.is_distributed else like.device
        return (
            Tensor(self._rope_table, device=CPU()).cast(like.dtype).to(target)
        )

    def project_context(self, taps: Tensor) -> Tensor:
        """Projects the concatenated target taps to the context stream."""
        return self.context_norm(self.context_proj(taps))

    def materialize(
        self,
        ctx: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes every layer's context latent for the verified rows."""
        rope = self._rope(ctx)
        for layer in self.layers:
            layer.self_attn.materialize(
                ctx, kv_collection, input_row_offsets, rope
            )

    def forward_block(
        self,
        embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        buffer_length: Tensor,
    ) -> Tensor:
        """Runs the block's rows through the layers and the final norm.

        The prefill plan is computed once here and shared by every layer,
        the target's own rule: it reads only cache lengths and row offsets,
        which no layer changes. ``buffer_length`` is the host extent the
        caller selected for this block -- the block's real extent depends
        on the verify's device-side commit, so the caller passes what
        :func:`~.mla_extent.k3_spec_mla_extent` selects, never a value it
        cannot prove covers the plan's own.
        """
        h = embeds
        rope = self._rope(h)
        prefill_plan = kimi_k3_mla_prefill_plan(
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            buffer_size=self.mla_buffer_size,
            buffer_length=buffer_length,
        )
        for layer in self.layers:
            h = layer(h, kv_collection, input_row_offsets, rope, prefill_plan)
        return self.final_norm(h)

    def forward(
        self,
        embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        buffer_length: Tensor,
    ) -> Tensor:
        """Alias for :meth:`forward_block`."""
        return self.forward_block(
            embeds, kv_collection, input_row_offsets, buffer_length
        )

    def draft_chain(
        self,
        logits: Tensor,
        anchors: Tensor,
        sampler: Callable[[Tensor, int], Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draws ``block_size`` draft tokens per request, one slot at a time.

        Slot ``i``'s logits get the bigram bias of the token drawn at slot
        ``i - 1`` -- the anchor at slot 0, whose own hidden is the first
        proposal; the loop unrolls over the static ``block_size``.

        Args:
            logits: ``[b * block_size, vocab]`` float32, :meth:`logits`'
                output.
            anchors: ``[b]`` int, each request's committed token.
            sampler: Draws slot ``i``'s ``[b]`` tokens from its ``[b,
                vocab]`` biased logits; greedy argmax when ``None``.

        Returns:
            ``([b, block_size]`` int32 draft ids, ``[b, block_size, vocab]``
            float32 biased logits).
        """
        sample = _greedy if sampler is None else sampler
        batch = anchors.shape[0]
        vocab = logits.shape[-1]
        logits = F.reshape(
            F.rebind(logits, [batch * self.block_size, vocab]),
            [batch, self.block_size, vocab],
        )
        current = anchors.cast(DType.int32)
        drafted: list[Tensor] = []
        biased: list[Tensor] = []
        for slot in range(self.block_size):
            step = logits[:, slot] + self.markov_head.bias_float32(current)
            biased.append(step)
            current = sample(step, slot).cast(DType.int32)
            drafted.append(current)
        return F.stack(drafted, axis=1), F.stack(biased, axis=1)

    def logits(self, hidden: Tensor, head_weight: Tensor) -> Tensor:
        """Scores the block through the target's LM head, on device 0.

        The GEMM runs in the weights' bfloat16 and widens after, so the
        draft logits round the way the target's own head does; the
        vocabulary-axis allgather splits the same element-wise dot products
        across ranks, changing no element's rounding.

        Args:
            hidden: :meth:`forward_block`'s output, ``[rows, hidden]``.
            head_weight: The target's ``[vocab, hidden]`` LM head weight,
                row-sharded over the vocabulary under TP.

        Returns:
            ``[rows, vocab]`` float32 on the first device.
        """
        out = hidden @ head_weight.T
        if out.is_distributed and any(
            isinstance(placement, Sharded) and placement.axis == 0
            for placement in head_weight.placements
        ):
            out = F.allgather(out, tensor_axis=-1).local_shards[0]
        return out.cast(DType.float32)


def tensor_parallel_k3_dspark(draft: K3DSpark, mesh: DeviceMesh) -> K3DSpark:
    """Shards the drafter by head along the ``tp`` mesh axis.

    The context projection, the embedding, every norm, and the Markov head
    stay replicated: the taps and block ids arrive replicated and the serial
    chain runs on one device. ``q_b_proj`` and ``kv_b_proj`` shard on their
    output, ``o_proj`` and the MLP's ``down_proj`` on their input, leaving
    partial sums the layers reduce.
    """
    tp = mesh.axis_size(TP)
    if tp == 1:
        return draft
    check_divisible("DSpark head count", draft.config.num_attention_heads, tp)
    check_divisible(
        "DSpark intermediate size", draft.config.intermediate_size, tp
    )
    row_mapping = NamedMapping(mesh, (TP, None))
    for layer in draft.layers:
        layer.self_attn.q_b_proj.weight = layer.self_attn.q_b_proj.weight.to(
            row_mapping
        )
        layer.self_attn.kv_b_proj.weight = layer.self_attn.kv_b_proj.weight.to(
            row_mapping
        )
        layer.self_attn.o_proj.weight = layer.self_attn.o_proj.weight.to(
            NamedMapping(mesh, (None, TP))
        )
        layer.mlp.gate_proj.weight = layer.mlp.gate_proj.weight.to(row_mapping)
        layer.mlp.up_proj.weight = layer.mlp.up_proj.weight.to(row_mapping)
        layer.mlp.down_proj.weight = layer.mlp.down_proj.weight.to(
            NamedMapping(mesh, (None, TP))
        )
    return draft
