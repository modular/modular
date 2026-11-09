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
"""Muse Spark's DSpark drafter on ModuleV3.

The drafter follows vLLM's ``vllm/models/muse_spark/common/dspark.py``:

- ``fc`` joins the target's tapped hidden states and ``hidden_norm``
  normalizes them. That is the draft's context: each layer projects it to K
  and V through its own input norm and writes them into the draft's cache at
  the context rows' positions. It never runs through the layers.
- The block, the committed token then mask tokens, embeds through the
  target's ``embed_tokens`` and runs through Muse decoder layers whose MoE is
  a dense SwiGLU MLP, on an fp32 residual stream, with causal sliding-window
  attention over the cached context and the block.
- Every slot drafts, the anchor's included. Its logits are the draft's own
  ``lm_head`` plus the Markov bias of the token drawn before it.

RoPE rotates interleaved pairs: the target's loader reorders its q and k rows
for that, the draft's checkpoint is stored that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding, Linear, Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kernels import (
    rope_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams
from max.nn.sequential import ModuleList
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated

from ..muse_spark.layers.attention import MuseSparkAttention
from ..muse_spark.layers.norms import (
    CenteredRMSNorm,
    InputScaledRMSNorm,
    WeightlessRMSNorm,
)
from ..muse_spark.layers.residual_gate import MuseSparkResidualGate
from ..speculators_common.draft_config import DSparkSpeculatorsDraftConfig

# The draft fields the layers below implement; any other value changes the
# math (vLLM's MuseSpark1xAttention and MuseSpark1xDecoderLayer for a draft).
_IMPLEMENTED = {
    "use_qk_norm": True,
    "use_attn_o_gate": True,
    "use_attn_o_norm": True,
    "use_residual_gate": True,
    "use_rope": True,
    "every_n_layers_nope": None,
    "attn_temperature_tuning_floor_scale": None,
    "norm_affine": True,
    "norm_zero_centered_gamma": True,
    "post_norm_affine": False,
    "post_norm_gain_center_type": "one",
    "use_ffn_learnable_scales": True,
    "residual_dtype": "fp32",
    "enable_confidence_head": True,
}


@dataclass(frozen=True)
class MuseSparkDraftConfig:
    """The drafter's hyperparameters, from its checkpoint config."""

    spec: DSparkSpeculatorsDraftConfig
    """The fields every speculators-format DSpark draft has."""
    post_norm_eps: float
    qk_scale_factor: float
    residual_gate_temperature: float
    output_multiplier: float
    final_logit_softcapping: float
    confidence_with_markov: bool
    target_num_hidden_layers: int

    # The fields the spec-decode pairing reads off a draft's config.
    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        """The target layers whose outputs the drafter reads."""
        return self.spec.target_layer_ids

    @property
    def num_target_layers(self) -> int:
        """The target model's layer count."""
        return self.target_num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """The drafter's hidden width."""
        return self.spec.hidden_size

    @property
    def vocab_size(self) -> int:
        """The drafter's vocabulary size."""
        return self.spec.vocab_size

    @property
    def mask_token_id(self) -> int:
        """The id that fills the block's draft slots."""
        return self.spec.mask_token_id

    @property
    def block_size(self) -> int:
        """The drafter's block width."""
        return self.spec.block_size

    @classmethod
    def from_huggingface_config(cls, hf: Any) -> MuseSparkDraftConfig:
        """Parses the ``muse_spark_dspark`` config.

        Raises:
            ValueError: If the config describes a draft these layers do not
                build.
        """
        spec = DSparkSpeculatorsDraftConfig.from_huggingface_config(hf)
        unsupported = {
            key: getattr(hf, key, None)
            for key, value in _IMPLEMENTED.items()
            if getattr(hf, key, None) != value
        }
        if (getattr(hf, "moe", None) or {}).get("is_enabled"):
            unsupported["moe"] = hf.moe
        if spec.has_d2t:
            unsupported["draft_vocab_size"] = spec.draft_vocab_size
        if list(hf.global_attn_cfg) != [spec.sliding_window]:
            unsupported["global_attn_cfg"] = hf.global_attn_cfg
        if unsupported:
            raise ValueError(
                f"The Muse Spark DSpark draft does not support {unsupported}."
            )
        # vLLM's MetapArgs defaults fill the keys a config leaves out.
        metap = {
            "use_metap": False,
            "metap_mode": "mup",
            "base_width": 256.0,
            "router_has_one_inf_dim": True,
            **(getattr(hf, "metap", None) or {}),
        }
        output_multiplier = 1.0
        if metap["use_metap"] and metap["router_has_one_inf_dim"]:
            exponents = {"mup": -1.0, "sp": -0.5}
            if metap["metap_mode"] not in exponents:
                raise ValueError(
                    f"Unknown metap_mode {metap['metap_mode']!r} in the"
                    " Muse Spark DSpark draft config."
                )
            output_multiplier = (
                spec.hidden_size / metap["base_width"]
            ) ** exponents[metap["metap_mode"]]
        # The drafter's config does not carry these; vLLM's Muse Spark model
        # hardcodes them for the draft layers.
        return cls(
            spec=spec,
            post_norm_eps=hf.rms_post_norm_eps or spec.rms_norm_eps,
            qk_scale_factor=(
                3.87
                if hf.use_sqrt_attn_scaling
                else 43.7840518911 / math.sqrt(spec.head_dim)
            ),
            residual_gate_temperature=0.3,
            output_multiplier=output_multiplier,
            final_logit_softcapping=hf.output_soft_cap_temp,
            confidence_with_markov=hf.confidence_head_with_markov,
            target_num_hidden_layers=hf.target_num_hidden_layers,
        )


class MuseSparkDSparkAttention(MuseSparkAttention):
    """The target's attention, plus the input norm and the context's K/V path."""

    def __init__(
        self,
        config: MuseSparkDraftConfig,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
        layer_idx: int,
    ) -> None:
        spec = config.spec
        super().__init__(
            rope=rope,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            hidden_size=spec.hidden_size,
            kv_params=kv_params,
            layer_idx_in_cache=layer_idx,
            is_sliding=True,
            norm_eps=spec.rms_norm_eps,
            qk_scale_factor=config.qk_scale_factor,
            local_window_size=spec.sliding_window,
        )
        self.pre_qkv_proj_norm = CenteredRMSNorm(
            spec.hidden_size, spec.rms_norm_eps
        )

    def materialize(
        self,
        context: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes the K and V of the context rows into the cache."""
        x = self.pre_qkv_proj_norm(context)
        heads = [-1, self.num_key_value_heads, self.head_dim]
        k = self._head_norm(self.k_proj(x), self.num_key_value_heads)
        k = k.reshape(heads)
        # The rows sit at their cache lengths onward, as attention reads them.
        k = rope_ragged(
            k.cast(DType.float32),
            input_row_offsets,
            kv_collection.cache_lengths,
            self.rope.freqs_cis,
            interleaved=self.rope.interleaved,
            output_dtype=self.kv_params.dtype,
        ).rebind_mapping(k.mapping)
        layer_idx = F.constant(
            self.layer_idx_in_cache, DType.uint32, device=CPU()
        )
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
        store_v_cache_ragged(
            kv_collection,
            self.v_proj(x).reshape(heads),
            input_row_offsets,
            layer_idx,
        )


class MuseSparkDSparkLayer(Module[..., Tensor]):
    """A Muse decoder layer with a dense MLP, on an fp32 residual stream."""

    def __init__(
        self,
        config: MuseSparkDraftConfig,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
        layer_idx: int,
    ) -> None:
        super().__init__()
        spec = config.spec
        hidden, temperature = spec.hidden_size, config.residual_gate_temperature
        self.self_attn = MuseSparkDSparkAttention(
            config, rope, kv_params, layer_idx
        )
        self.post_attn_norm = WeightlessRMSNorm(config.post_norm_eps)
        self.add_attn = MuseSparkResidualGate(hidden, temperature)
        self.pre_ffn_norm = CenteredRMSNorm(hidden, spec.rms_norm_eps)
        self.feed_forward = MLP(hidden, spec.intermediate_size)
        self.post_ffn_norm = InputScaledRMSNorm(hidden, config.post_norm_eps)
        self.add_ffn = MuseSparkResidualGate(hidden, temperature)

    def forward(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the drafter layer on ``h``."""
        # Each trained input norm reads the stream rounded to the compute
        # dtype, as the reference does.
        # Under TP o_proj and down_proj leave partial sums; the norms need
        # whole rows.
        whole = DeviceMapping(h.mesh, (Replicated(),))
        attn = self.self_attn(
            self.self_attn.pre_qkv_proj_norm(h.cast(DType.bfloat16)),
            kv_collection,
            input_row_offsets=input_row_offsets,
        ).to(whole)
        h = self.add_attn(h, self.post_attn_norm(attn))
        ffn = self.feed_forward(self.pre_ffn_norm(h.cast(DType.bfloat16)))
        return self.add_ffn(h, self.post_ffn_norm(ffn.to(whole)))


class ConfidenceHead(Module[..., Tensor]):
    """Each slot's acceptance probability, in fp32."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = Linear(width, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """The fp32 sigmoid confidence of each row."""
        w, b = self.proj.weight, self.proj.bias
        assert isinstance(b, Tensor)
        logit = x.cast(DType.float32) @ w.cast(DType.float32).T
        return F.sigmoid(logit + b.cast(DType.float32)).squeeze(-1)


class MarkovHead(Module[..., Tensor]):
    """The low-rank bias ``markov_w2(markov_w1[previous token])``."""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = Embedding(vocab_size, dim=rank)
        self.markov_w2 = Linear(rank, vocab_size, bias=False)

    def forward(self, embedded: Tensor) -> Tensor:
        """The Markov bias logits of the embedded previous tokens."""
        return self.markov_w2(embedded)


class MuseSparkDSpark(Module[..., Tensor]):
    """The drafter: context projection, layers and heads."""

    # Its head is its own; the block embeds through the target's table.
    has_own_lm_head = True

    def __init__(
        self,
        config: MuseSparkDraftConfig,
        kv_params: KVCacheParams,
        *,
        max_seq_len: int,
        device: Any,
    ) -> None:
        super().__init__()
        spec = config.spec
        hidden = spec.hidden_size
        self.config = config
        self.rope = RotaryEmbedding(
            dim=hidden,
            n_heads=spec.num_attention_heads,
            theta=spec.rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            head_dim=spec.head_dim,
            interleaved=True,
        )
        self.fc = Linear(spec.num_context_features * hidden, hidden, bias=False)
        self.hidden_norm = CenteredRMSNorm(hidden, spec.rms_norm_eps)
        self.layers = ModuleList(
            [
                MuseSparkDSparkLayer(config, self.rope, kv_params, i)
                for i in range(spec.num_hidden_layers)
            ]
        )
        self.norm = CenteredRMSNorm(hidden, spec.rms_norm_eps)
        self.lm_head = Linear(hidden, spec.vocab_size, bias=False)
        self.markov_head = MarkovHead(spec.vocab_size, spec.markov_rank)
        width = hidden + (
            spec.markov_rank if config.confidence_with_markov else 0
        )
        self.confidence_head = ConfidenceHead(width)

    def prepare_freq_cis(self, mesh: DeviceMesh) -> None:
        """Places the RoPE table on ``mesh`` in bfloat16."""
        self.rope.freqs_cis = self.rope.freqs_cis.cast(DType.bfloat16).to(mesh)

    def materialize(
        self,
        taps: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes every layer's K and V of the context rows.

        Args:
            taps: The target's tapped hidden states, ``[rows, taps * hidden]``.
            kv_collection: The draft's cache at the context's position.
            input_row_offsets: The context rows' ragged offsets.
        """
        context = self.hidden_norm(self.fc(taps.cast(DType.bfloat16)))
        for layer in self.layers:
            layer.self_attn.materialize(
                context, kv_collection, input_row_offsets
            )

    def forward(
        self,
        embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the embedded block; returns its final-normed rows."""
        h = embeds.cast(DType.float32)
        for layer in self.layers:
            h = layer(h, kv_collection, input_row_offsets)
        return self.norm(h.cast(DType.bfloat16))

    def markov_embed(self, previous: Tensor) -> Tensor:
        """The rank-``r`` embedding of each row's previous token."""
        return F.gather(self.markov_head.markov_w1.weight, previous, axis=0)

    def slot_logits(self, base: Tensor, markov_embed: Tensor) -> Tensor:
        """One slot's fp32 logits: the base, the Markov bias, the transform."""
        logits = (base + self.markov_head(markov_embed)).cast(DType.float32)
        cap = self.config.final_logit_softcapping
        return F.tanh(logits * self.config.output_multiplier / cap) * cap

    def confidence(self, hidden: Tensor, markov_embed: Tensor) -> Tensor:
        """Each slot's predicted acceptance probability."""
        if self.config.confidence_with_markov:
            hidden = F.concat([hidden, markov_embed], axis=-1)
        return self.confidence_head(hidden)
