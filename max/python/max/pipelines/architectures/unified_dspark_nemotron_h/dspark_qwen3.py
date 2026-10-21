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
"""The DSpark draft for a Nemotron-H target, a Qwen3-style block draft.

The draft is the DFlash backbone of Model Optimizer's ``DFlashModule`` and the
DSpark Markov head on top of it:

- The target's tapped hidden states are concatenated, projected by ``fc`` and
  normalized by ``hidden_norm``. The result is the draft's context. It is
  never run through the layers: each layer projects it to K and V and writes
  them into the draft's own KV cache at the context rows' positions.
- The block is ``[anchor, mask x (block_size - 1)]``. It runs once through
  the layers, a causal sliding-window attention over the cached context and
  the block's own K and V, with a learned sink logit per head.
- The block's hidden states give the base logits through the *target's*
  ``lm_head`` (the draft ships none). The Markov head then adds
  ``markov_w2(markov_w1[previous token])`` to the logits of each position in
  turn, the previous token being the sampled one, and the anchor for the first.

Every module is a ModuleV3 module under the checkpoint's own names, so the
checkpoint loads as stored. The MLPs, ``fc`` and ``markov_w2`` are NVFP4
weight-only in the checkpoint and run W4A4 on the block-scaled matmul the
target's NVFP4 layers use; attention, the norms and ``markov_w1`` stay BF16.

This is a second implementation next to ``dspark_draft/dspark_speculators_draft``
rather than an extension of it. That draft is built from graph-API layers for
one device and carries its own ``lm_head`` and ``d2t``. This one needs ModuleV3
layers for tensor parallelism, NVFP4 W4A4 linears and this checkpoint's names,
and shares the target's ``lm_head``. A change to the attention mask, sink
handling or Markov bias must be made in both until they are merged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    store_k_cache_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import col_parallel, row_parallel
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import fused_qk_ragged_rope as _fused_qk_ragged_rope
from max.nn.kernels import store_v_cache_ragged as _store_v_cache_ragged
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.nemotron_h_modulev3.layers.quantized import (
    NVFP4Linear,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.sharding import (
    shard_axis,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    Parallelism,
)

store_v_cache_ragged = F.functional(_store_v_cache_ragged)
fused_qk_ragged_rope = F.functional(_fused_qk_ragged_rope)

__all__ = [
    "NVFP4_DRAFT_ALGO",
    "DSparkQwen3",
    "DSparkQwen3DraftConfig",
    "draft_nvfp4_modules",
    "markov_bias",
    "parse_dspark_draft_config",
]

NVFP4_DRAFT_ALGO = "W4A16_NVFP4"
"""The modelopt algorithm the draft's quantized linears are stored with."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Reads ``name`` from a Hugging Face config object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass(frozen=True)
class DSparkQwen3DraftConfig:
    """The DSpark draft's hyperparameters, from its checkpoint config."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    sliding_window: int
    max_position_embeddings: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    """The target layers whose *outputs* the draft reads, as the checkpoint
    names them. Tapping a layer's output is tapping the next layer's input."""
    markov_rank: int
    sample_from_anchor: bool
    """Whether the anchor slot of the block predicts a draft token.

    ``False`` is the DFlash layout: the block's first slot holds the committed
    token and predicts nothing, so a block of ``block_size`` carries
    ``block_size - 1`` drafts."""
    quantized: bool
    """Whether the checkpoint stores its linears NVFP4 weight-only."""

    @property
    def num_context_features(self) -> int:
        """How many target taps feed the ``fc`` projection."""
        return len(self.target_layer_ids)

    @property
    def num_drafts(self) -> int:
        """How many tokens a block proposes."""
        return (
            self.block_size if self.sample_from_anchor else self.block_size - 1
        )

    @property
    def tap_layer_inputs(self) -> tuple[int, ...]:
        """The target layers whose *inputs* the target taps.

        The checkpoint names layer outputs and the target taps layer inputs,
        and a layer's output is the next layer's input.
        """
        return tuple(i + 1 for i in self.target_layer_ids)


def parse_dspark_draft_config(hf_config: Any) -> DSparkQwen3DraftConfig:
    """Reads the draft's checkpoint config.

    Raises:
        ValueError: If the config describes a draft this implementation does
            not build.
    """
    dflash = _get(hf_config, "dflash_config", None) or {}

    def field(name: str, default: Any = None) -> Any:
        value = _get(dflash, name, None)
        return _get(hf_config, name, default) if value is None else value

    markov_type = str(_get(hf_config, "markov_head_type", "vanilla"))
    if markov_type != "vanilla":
        raise ValueError(
            "The DSpark draft's Markov head must be 'vanilla', got"
            f" {markov_type!r}."
        )
    fixup = _get(hf_config, "dspark_fixup_head_type", "markov")
    if fixup != "markov":
        raise ValueError(
            f"The DSpark fixup head must be 'markov', got {fixup!r}."
        )
    rank = _get(hf_config, "dspark_markov_rank", None) or _get(
        hf_config, "markov_rank", 0
    )
    if not rank:
        raise ValueError("The DSpark draft config declares no markov_rank.")
    if _get(hf_config, "enable_confidence_head", False):
        raise ValueError(
            "The DSpark confidence head gates dynamic proposal lengths, which"
            " MAX does not use; use a draft without one."
        )
    if _get(hf_config, "hidden_act", "silu") != "silu":
        raise ValueError("The DSpark draft's MLP must use SiLU.")
    if not field("causal", True) or not _get(
        hf_config, "dflash_query_causal", True
    ):
        raise ValueError("The DSpark draft's block attention must be causal.")
    if not field("attention_sink_bias", False):
        raise ValueError("The DSpark draft must have attention sink biases.")
    window = field("swa_window_size", _get(hf_config, "sliding_window", None))
    layer_types = list(_get(hf_config, "layer_types", None) or [])
    if not window or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            "The DSpark draft must use sliding-window attention on every"
            f" layer, got window={window} and layer_types={layer_types}."
        )
    if _get(hf_config, "attention_bias", False):
        raise ValueError("The DSpark draft's attention has no biases.")
    if field("sample_from_anchor", False):
        raise ValueError(
            "The DSpark draft's anchor slot must not predict"
            " (sample_from_anchor=false): the proposer drops it and proposes"
            " block_size - 1 tokens."
        )
    mask_token_id = field("mask_token_id", None)
    target_layer_ids = field("target_layer_ids", None)
    block_size = field("block_size", None)
    if mask_token_id is None or not target_layer_ids or not block_size:
        raise ValueError(
            "The DSpark draft config must declare mask_token_id,"
            " target_layer_ids and block_size."
        )
    quant = _get(hf_config, "quantization_config", None)
    quantized = quant is not None
    if quantized:
        algo = _get(quant, "quant_algo", None)
        group = _get(quant, "group_size", None)
        if algo != NVFP4_DRAFT_ALGO or group != 16:
            raise ValueError(
                "The DSpark draft supports only unquantized weights or"
                f" {NVFP4_DRAFT_ALGO} with group size 16, got {algo} with"
                f" group size {group}."
            )
    head_dim = _get(hf_config, "head_dim", None) or (
        _get(hf_config, "hidden_size") // _get(hf_config, "num_attention_heads")
    )
    rope_parameters = _get(hf_config, "rope_parameters", None) or {}
    rope_theta = _get(rope_parameters, "rope_theta", None) or _get(
        hf_config, "rope_theta", 10000.0
    )
    return DSparkQwen3DraftConfig(
        hidden_size=int(_get(hf_config, "hidden_size")),
        intermediate_size=int(_get(hf_config, "intermediate_size")),
        num_hidden_layers=int(_get(hf_config, "num_hidden_layers")),
        num_attention_heads=int(_get(hf_config, "num_attention_heads")),
        num_key_value_heads=int(_get(hf_config, "num_key_value_heads")),
        head_dim=int(head_dim),
        rms_norm_eps=float(_get(hf_config, "rms_norm_eps", 1e-6)),
        vocab_size=int(_get(hf_config, "vocab_size")),
        rope_theta=float(rope_theta),
        sliding_window=int(window),
        max_position_embeddings=int(
            _get(hf_config, "max_position_embeddings", 1 << 20)
        ),
        block_size=int(block_size),
        mask_token_id=int(mask_token_id),
        target_layer_ids=tuple(int(i) for i in target_layer_ids),
        markov_rank=int(rank),
        sample_from_anchor=False,
        quantized=quantized,
    )


def draft_nvfp4_modules(
    config: DSparkQwen3DraftConfig,
) -> dict[str, Parallelism]:
    """Returns the draft's NVFP4 linear modules and how each is split.

    Every linear but attention's is quantized. The Markov head's ``markov_w2``
    and ``fc`` are replicated: the head runs on one device, and ``fc`` reads
    every tap, which the target replicates.
    """
    if not config.quantized:
        return {}
    modules = {
        "fc": Parallelism.REPLICATED,
        "markov_head.markov_w2": Parallelism.REPLICATED,
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}.mlp"
        modules[f"{prefix}.gate_proj"] = Parallelism.COLUMN
        modules[f"{prefix}.up_proj"] = Parallelism.COLUMN
        modules[f"{prefix}.down_proj"] = Parallelism.ROW
    return modules


def _linear(
    config: DSparkQwen3DraftConfig,
    quantized: Mapping[str, Parallelism],
    name: str,
    in_dim: int,
    out_dim: int,
    parallelism: Parallelism,
    num_devices: int,
) -> Linear | NVFP4Linear:
    """Builds the linear ``name``, NVFP4 if the checkpoint stores it so."""
    if name in quantized:
        return NVFP4Linear(in_dim, out_dim, quantized[name], num_devices)
    linear = Linear(in_dim, out_dim, bias=False)
    match parallelism:
        case Parallelism.COLUMN:
            return col_parallel(linear)
        case Parallelism.ROW:
            return row_parallel(linear)
    return linear


class DSparkAttention(Module[..., Tensor]):
    """Causal sliding-window GQA with a sink logit per head.

    The layer's K and V reach the cache by two paths that must agree: the
    context rows' (:meth:`materialize`) and the block's own
    (:meth:`forward`).
    """

    def __init__(
        self,
        config: DSparkQwen3DraftConfig,
        kv_params: KVCacheParams,
        layer_idx: int,
        rope: RotaryEmbedding,
    ) -> None:
        self.config = config
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.rope = rope
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_proj = col_parallel(
            Linear(hidden, self.n_heads * self.head_dim, bias=False)
        )
        self.k_proj = col_parallel(
            Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        )
        self.v_proj = col_parallel(
            Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        )
        self.o_proj = row_parallel(
            Linear(self.n_heads * self.head_dim, hidden, bias=False)
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attention_sink_bias = shard_axis(Tensor.zeros([self.n_heads]))

    def _layer(self) -> Tensor:
        return F.constant(self.layer_idx, DType.uint32, device=CPU())

    def _freqs(self, like: Tensor) -> Tensor:
        return F.cast(self.rope.freqs_cis, like.dtype).to(like.mesh)

    def _store_kv(
        self,
        k: Tensor,
        v: Tensor,
        q: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Writes K and V of the rows to the cache and ropes ``q`` and K.

        The rope kernel rotates the cached K of the rows together with a query,
        so a caller with no query passes K in its place and drops the result.
        """
        layer = self._layer()
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer)
        store_v_cache_ragged(kv_collection, v, input_row_offsets, layer)
        return fused_qk_ragged_rope(
            self.kv_params,
            q,
            input_row_offsets,
            kv_collection,
            self._freqs(q),
            layer,
            interleaved=False,
        )

    def _kv(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Projects ``x`` to the layer's per-head K, normalized, and V."""
        heads = [-1, self.n_kv_heads, self.head_dim]
        k = self.k_norm(self.k_proj(x).reshape(heads))
        v = self.v_proj(x).reshape(heads)
        return k, v

    def materialize(
        self,
        ctx: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes the K and V of the context rows into the cache.

        The context is the draft's own input from the target and is not run
        through the layer's norms or projections beyond K and V.
        """
        k, v = self._kv(ctx)
        sharded = NamedMapping(k.mesh, (None, TP, None))
        k = k.rebind_mapping(sharded)
        v = v.rebind_mapping(sharded)
        self._store_kv(k, v, k, kv_collection, input_row_offsets)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        q = self.q_norm(
            self.q_proj(x).reshape([-1, self.n_heads, self.head_dim])
        )
        k, v = self._kv(x)
        kv_mapping = NamedMapping(k.mesh, (None, TP, None))
        q = q.rebind_mapping(kv_mapping)
        k = k.rebind_mapping(kv_mapping)
        v = v.rebind_mapping(kv_mapping)
        q = self._store_kv(k, v, q, kv_collection, input_row_offsets)
        q = q.rebind_mapping(kv_mapping)
        attn = flash_attention_ragged(
            self.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=self._layer(),
            mask_variant=MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
            scale=1.0 / math.sqrt(self.head_dim),
            local_window_size=self.config.sliding_window,
            sink_weights=self.attention_sink_bias,
        ).rebind_mapping(kv_mapping)
        return self.o_proj(attn.reshape([-1, self.n_heads * self.head_dim]))


class DSparkMLP(Module[[Tensor], Tensor]):
    """A SiLU-gated MLP, NVFP4 as the checkpoint stores it."""

    def __init__(
        self,
        config: DSparkQwen3DraftConfig,
        quantized: Mapping[str, Parallelism],
        layer_idx: int,
        num_devices: int,
    ) -> None:
        prefix = f"layers.{layer_idx}.mlp"
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = _linear(
            config,
            quantized,
            f"{prefix}.gate_proj",
            hidden,
            inner,
            Parallelism.COLUMN,
            num_devices,
        )
        self.up_proj = _linear(
            config,
            quantized,
            f"{prefix}.up_proj",
            hidden,
            inner,
            Parallelism.COLUMN,
            num_devices,
        )
        self.down_proj = _linear(
            config,
            quantized,
            f"{prefix}.down_proj",
            inner,
            hidden,
            Parallelism.ROW,
            num_devices,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkLayer(Module[..., Tensor]):
    """A pre-norm Qwen3 decoder layer over the block."""

    def __init__(
        self,
        config: DSparkQwen3DraftConfig,
        quantized: Mapping[str, Parallelism],
        kv_params: KVCacheParams,
        layer_idx: int,
        rope: RotaryEmbedding,
        num_devices: int,
    ) -> None:
        self.self_attn = DSparkAttention(config, kv_params, layer_idx, rope)
        self.mlp = DSparkMLP(config, quantized, layer_idx, num_devices)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        # Row-parallel layers leave partial sums; reduce before the residual.
        attn = self.self_attn(
            self.input_layernorm(h), kv_collection, input_row_offsets
        )
        h = h + attn.to(h.mapping)
        return h + self.mlp(self.post_attention_layernorm(h)).to(h.mapping)


class MarkovHead(Module[..., Tensor]):
    """The low-rank transition bias ``markov_w2(markov_w1[previous token])``."""

    def __init__(
        self,
        config: DSparkQwen3DraftConfig,
        quantized: Mapping[str, Parallelism],
        num_devices: int,
    ) -> None:
        self.markov_w1 = Embedding(config.vocab_size, dim=config.markov_rank)
        self.markov_w2 = _linear(
            config,
            quantized,
            "markov_head.markov_w2",
            config.markov_rank,
            config.vocab_size,
            Parallelism.REPLICATED,
            num_devices,
        )


def markov_bias(head: MarkovHead, previous: Tensor) -> Tensor:
    """Returns the bias of each row's next-token logits, on device 0.

    The head is replicated and runs once, on the first device's copy, because
    the sampler that picks each position's token runs there and the
    positions depend on one another.

    Args:
        head: The Markov head.
        previous: The ``[batch]`` previous token ids on device 0.

    Returns:
        The ``[batch, vocab]`` bias, as the weights' dtype.
    """
    embed_weight = head.markov_w1.weight.local_shards[0]
    embedded = F.gather(embed_weight, previous, axis=0)
    w2 = head.markov_w2
    if isinstance(w2, NVFP4Linear):
        return Tensor.from_graph_value(w2.matmul_on(TensorValue(embedded), 0))
    return embedded @ w2.weight.local_shards[0].T


class DSparkQwen3(Module[..., Tensor]):
    """The DSpark draft: context projection, layers and Markov head."""

    def __init__(
        self,
        config: DSparkQwen3DraftConfig,
        kv_params: KVCacheParams,
        *,
        num_devices: int,
        max_seq_len: int,
        device: Any,
    ) -> None:
        """Builds the draft.

        Args:
            config: The draft's hyperparameters.
            kv_params: The draft's KV cache leaf.
            num_devices: The tensor-parallel degree.
            max_seq_len: The longest sequence, which sizes the rope table.
            device: The device the rope table is built on.
        """
        n_kv_heads = config.num_key_value_heads
        if n_kv_heads % num_devices:
            raise ValueError(
                f"The DSpark draft has {n_kv_heads} KV heads, which"
                f" {num_devices} devices do not divide: each device holds"
                " whole KV heads."
            )
        if config.num_attention_heads % num_devices:
            raise ValueError(
                f"The DSpark draft has {config.num_attention_heads} attention"
                f" heads, which {num_devices} devices do not divide."
            )
        self.config = config
        quantized = draft_nvfp4_modules(config)
        self.quantized = quantized
        rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
        )
        self.fc = _linear(
            config,
            quantized,
            "fc",
            config.num_context_features * config.hidden_size,
            config.hidden_size,
            Parallelism.REPLICATED,
            num_devices,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The trained embedding of the mask token. The rest of the block's
        # embeddings are the target's.
        self.mask_embedding = Tensor.zeros([config.hidden_size])
        self.layers = ModuleList(
            [
                DSparkLayer(config, quantized, kv_params, i, rope, num_devices)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(config, quantized, num_devices)

    def project_context(self, taps: Tensor) -> Tensor:
        """Projects the concatenated target taps to the draft's context."""
        return self.hidden_norm(self.fc(taps))

    def materialize(
        self,
        ctx: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes every layer's K and V of the context rows into the cache."""
        for layer in self.layers:
            assert isinstance(layer, DSparkLayer)
            layer.self_attn.materialize(ctx, kv_collection, input_row_offsets)

    def forward_block(
        self,
        embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block, whose K and V land at the cache's lengths."""
        h = embeds
        for layer in self.layers:
            h = layer(h, kv_collection, input_row_offsets)
        return self.norm(h)

    def forward(
        self,
        embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        return self.forward_block(embeds, kv_collection, input_row_offsets)
