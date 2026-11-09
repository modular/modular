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
"""DiffusionGemma's two text models, which read one set of text weights.

Both are composed from the donor layers of the gemma4 ModuleV3 architecture.

- ``DiffusionGemmaEncoderTextModel``: causal forward that populates the
  paged KV cache (prompt prefill and accepted-canvas commits).
- ``DiffusionGemmaDecoderTextModel``: bidirectional forward over a
  ``canvas_length`` block that reads the encoder cache without committing
  to it, applies the self-conditioning block to its input embeddings, and
  computes the per-step sampling statistics (argmax, entropy, categorical
  sample, temperature-processed self-conditioning logits) on device.

Deltas vs the gemma4 ModuleV3 donor text model (each tagged at its
implementation site): the always-on parallel dense-MLP + MoE branch with the
fp32 router gate, the final logit softcapping ``tanh(l/30)*30`` on the
encoder, the noncausal decoder attention masks, and the self-conditioning
module.
"""

from __future__ import annotations

from typing import Any, TypeVar

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.transformer import ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib.role_contracts import BlockDiffusionOutputs
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .layers.attention import Gemma4Attention
from .layers.embedding import ScaledEmbedding
from .layers.moe import build_diffusion_gemma_moe
from .layers.rms_norm import Gemma4RMSNorm
from .layers.rotary_embedding import ProportionalRotaryEmbedding
from .layers.transformer_block import DiffusionGemmaTransformerBlock
from .model_config import DiffusionGemmaForBlockDiffusionConfig

_LAYER_TYPE_TO_KV_KEY = {
    "sliding_attention": "sliding_attention",
    "full_attention": "full_attention",
}


class DiffusionGemmaSelfConditioning(Module[[Tensor, Tensor], Tensor]):
    """Gated-MLP self-conditioning block (decoder-only weights).

    Mirrors HF ``DiffusionGemmaSelfConditioning``: the soft-embedding signal
    from the previous denoise step is pre-normed, passed through a
    gelu-tanh-gated MLP, added to the canvas input embeddings, and the sum is
    RMS-normalized without a learned scale.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.pre_norm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.post_norm = Gemma4RMSNorm(hidden_size, eps=eps, with_weight=False)
        self.gate_proj = Linear(
            in_dim=hidden_size, out_dim=intermediate_size, bias=False
        )
        self.up_proj = Linear(
            in_dim=hidden_size, out_dim=intermediate_size, bias=False
        )
        self.down_proj = Linear(
            in_dim=intermediate_size, out_dim=hidden_size, bias=False
        )

    def forward(self, inputs_embeds: Tensor, sc_signal: Tensor) -> Tensor:
        """Adds the self-conditioned signal to ``inputs_embeds``."""
        normed = self.pre_norm(sc_signal)
        gated = F.gelu(self.gate_proj(normed), approximate="tanh")
        sc = self.down_proj(gated * self.up_proj(normed))
        return self.post_norm(inputs_embeds + sc)


_OutputsT = TypeVar("_OutputsT")


class _DiffusionGemmaTextModelBase(Module[..., _OutputsT]):
    """Shared construction for the encoder and decoder text graphs.

    Builds embeddings, final norm, tied lm_head, RoPE tables, and the 30
    Gemma4 decoder layers (always-on parallel MoE branch, fp32 router gate).
    The two graphs differ only in their attention masks, so both register
    the same weight FQNs and load from the same adapted checkpoint dict.
    """

    def __init__(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        mesh: DeviceMesh,
        *,
        noncausal: bool,
    ) -> None:
        super().__init__()
        text_config = config.text_config
        self.config = config
        self.mesh = mesh
        # Under NVFP4, config.dtype is packed uint8; RoPE and attention run in
        # the unquantized dtype.
        self.dtype = (
            config.unquantized_dtype
            if text_config.quant_config is not None
            else config.dtype
        )
        device = mesh.devices[0]

        self.rope_local = RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.sliding_window_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            device=device,
            head_dim=text_config.head_dim,
            interleaved=False,
        )
        self.rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.global_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            device=device,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=text_config.global_rope_scaling,
        )

        self.embed_tokens = ScaledEmbedding(
            text_config.vocab_size,
            dim=text_config.hidden_size,
            embed_scale=text_config.hidden_size**0.5,
        )
        self.norm = Gemma4RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = ColumnParallelLinear(
                in_dim=text_config.hidden_size,
                out_dim=text_config.vocab_size,
                bias=False,
            )
        self.final_logit_softcapping = text_config.final_logit_softcapping

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type_key, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type_key] = kv_params_leaf

        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(text_config.num_hidden_layers):
            layer_type = text_config.layer_types[i]
            is_sliding = layer_type == "sliding_attention"
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1

            # The denoiser attends bidirectionally over the cache and canvas.
            mask_variant: MHAMaskVariant | None = None
            if noncausal:
                mask_variant = (
                    MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
                    if is_sliding
                    else MHAMaskVariant.NULL_MASK
                )
            # DiffusionGemma keeps every self_attn and the parallel dense
            # MLP unquantized under NVFP4; only the routed experts quantize.
            layers.append(
                DiffusionGemmaTransformerBlock(
                    attention=Gemma4Attention(
                        rope_global=self.rope_global,
                        rope_local=self.rope_local,
                        num_attention_heads=text_config.num_attention_heads,
                        num_key_value_heads=text_config.num_key_value_heads,
                        num_global_key_value_heads=(
                            text_config.num_global_key_value_heads
                        ),
                        attention_k_eq_v=text_config.attention_k_eq_v,
                        hidden_size=text_config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx_in_cache=layer_idx_in_cache,
                        is_sliding=is_sliding,
                        qk_norm_eps=text_config.rms_norm_eps,
                        local_window_size=text_config.sliding_window,
                        mask_variant=mask_variant,
                    ),
                    mlp=MLP(
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        activation_function=text_config.hidden_activation,
                    ),
                    moe_block=build_diffusion_gemma_moe(
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        eps=text_config.rms_norm_eps,
                        quant_config=text_config.quant_config,
                    ),
                    hidden_size=text_config.hidden_size,
                    rms_norm_eps=text_config.rms_norm_eps,
                )
            )
        self.layers = ModuleList(layers)
        self._layer_kv_key = [
            _LAYER_TYPE_TO_KV_KEY[text_config.layer_types[i]]
            for i in range(text_config.num_hidden_layers)
        ]
        self.return_logits = text_config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        """The tied lm_head, then the fp32 ``tanh(l / cap) * cap`` softcap.

        HF ``DiffusionGemmaForBlockDiffusion.forward`` computes its logits
        in float32; the softcap matches. Returns fp32 logits.
        """
        if self.tie_word_embeddings:
            outputs = h @ self.embed_tokens.weight.T
            outputs = F.allgather(outputs, tensor_axis=-1)
        else:
            assert self.lm_head is not None
            outputs = self.lm_head(h)
        outputs = outputs.cast(DType.float32)
        cap = self.final_logit_softcapping
        assert cap is not None, "DiffusionGemma sets final_logit_softcapping"
        return F.tanh(outputs / cap) * cap

    def prepare_freq_cis(self) -> None:
        """Computes the rotary frequencies for the global and local rope."""
        self.rope_global.freqs_cis = self.rope_global.freqs_cis.cast(
            self.dtype
        ).to(self.mesh)
        self.rope_local.freqs_cis = self.rope_local.freqs_cis.cast(
            self.dtype
        ).to(self.mesh)

    def _run_layers(
        self,
        h: Tensor,
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        kv_by_type = {
            "sliding_attention": sliding_kv,
            "full_attention": global_kv,
        }
        for idx, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )
        return h


class DiffusionGemmaEncoderTextModel(
    _DiffusionGemmaTextModelBase[tuple[Tensor, ...]]
):
    """Causal text encoder: populates the paged KV cache.

    Same interface as the gemma4 ModuleV3 donor text model (including the
    multimodal-embedding merge inputs, fed empty for text-only requests).
    Logits pass through the final softcap so encoder-side probes compare
    directly against the HF reference.
    """

    def __init__(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__(config, mesh, noncausal=False)

    def forward(
        self,
        tokens: Tensor,
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> tuple[Tensor, ...]:
        """Runs the encoder over the tokens and image embeddings."""
        # Stage on every device with one collective each (a no-op on one).
        tokens = F.distributed_broadcast(tokens, self.mesh)
        input_row_offsets = F.distributed_broadcast(
            input_row_offsets, self.mesh
        )
        image_embeddings = F.distributed_broadcast(image_embeddings, self.mesh)
        image_token_indices = F.distributed_broadcast(
            image_token_indices, self.mesh
        )
        h = self.embed_tokens(tokens)
        self.prepare_freq_cis()

        # Image embeddings arrive pre-scaled, so merge them after the embedding
        # scale. A text-only batch has no indices and is a no-op.
        h = F_merge_multimodal_embeddings(
            h, image_embeddings.cast(h.dtype), image_token_indices
        )

        h = self._run_layers(h, sliding_kv, global_kv, input_row_offsets)

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))

        logits: Tensor | None = None
        offsets: Tensor | None = None
        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = F.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=CPU(),
                dtype=DType.int64,
            )
            offsets = (
                input_row_offsets[1:].unsqueeze(-1) - return_n_logits_range
            )
            last_indices = offsets.reshape((-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = F.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=CPU(),
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)
        return ret_val


class DiffusionGemmaDecoderTextModel(
    _DiffusionGemmaTextModelBase[BlockDiffusionOutputs]
):
    """Bidirectional canvas denoiser with on-device sampling statistics.

    One call = one denoise step (HF ``_denoising_step`` minus the host-side
    acceptance/renoise logic):

    1. Canvas embeddings + self-conditioning signal (soft embeddings of the
       previous step's processed logits; zeroed via ``sc_enabled`` on the
       first step).
    2. 30 layers with noncausal attention over [encoder cache | canvas].
    3. Tied lm_head -> fp32 softcap -> divide by ``temperature`` (the linear
       schedule lives in the host loop).
    4. Outputs: processed logits as bf16 (next step's self-conditioning
       input, kept device-resident), per-token argmax, top-64
       probabilities/indices for host-side categorical sampling, and
       per-token full-vocab entropy.
    """

    def __init__(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__(config, mesh, noncausal=True)
        text_config = config.text_config
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            eps=text_config.rms_norm_eps,
        )

    def forward(
        self,
        canvas_tokens: Tensor,
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        input_row_offsets: Tensor,
        sc_logits: Tensor,
        sc_enabled: Tensor,
        temperature: Tensor,
    ) -> BlockDiffusionOutputs:
        """Runs the decoder over the canvas tokens."""
        h = self.embed_tokens(canvas_tokens)
        self.prepare_freq_cis()

        # Soft embeddings: softmax(prev processed logits, fp32) @ embed_table
        # * embed_scale, gated to zero on the first denoise step (HF passes
        # self_conditioning_logits=None there).
        embed_weight = self.embed_tokens.weight.to(self.mesh)
        soft_probs = F.softmax(sc_logits.cast(DType.float32))
        soft_embeds = soft_probs.cast(embed_weight.dtype) @ embed_weight
        soft_embeds = (
            soft_embeds
            * self.embed_tokens.embed_scale
            * sc_enabled.cast(embed_weight.dtype)
        )
        h = self.self_conditioning(h, soft_embeds)

        h = self._run_layers(h, sliding_kv, global_kv, input_row_offsets)

        logits = self._compute_logits(self.norm(h))  # fp32 [N, vocab]
        processed = logits / temperature.cast(DType.float32)

        probs = F.softmax(processed)
        argmax = processed.argmax(axis=-1)

        # Entropy of Categorical(logits=processed):
        # H = logsumexp(processed) - sum(p * processed).
        m = processed.max(axis=-1)
        lse = F.log(F.sum(F.exp(processed - m), axis=-1)) + m
        entropy = lse - F.sum(probs * processed, axis=-1)

        # Top-k probabilities for host-side categorical sampling. The exact
        # inverse-CDF needed ops.cumsum over [N, vocab] fp32, which has no
        # GPU kernel (KERN-1095) and cost two ~268 MB host transfers per
        # denoise step. Sampling from the renormalized top-64 instead ships
        # only [N, 64]; the truncated tail mass is negligible at the
        # schedule's temperatures, and the entropy/argmax used for
        # acceptance and stopping remain full-vocab and exact.
        topk_probs, topk_idx = F.top_k(probs, k=64, axis=-1)

        return BlockDiffusionOutputs(
            logits=processed.cast(DType.bfloat16),
            argmax=argmax.cast(DType.int64),
            top_k_probs=topk_probs.cast(DType.float32),
            top_k_indices=topk_idx.cast(DType.int64),
            entropy=entropy.cast(DType.float32),
        )


def unflatten_kv(
    kv_params: Any,
    variadic_args: tuple[Tensor, ...],
    mesh: DeviceMesh,
) -> tuple[PagedCacheValues, PagedCacheValues]:
    """Unflattens the two-cache KV tree a graph's trailing inputs carry.

    Each device owns a whole KV cache, so the shards are replicated over
    the mesh rather than sharded over it.
    """
    kv_inputs = iter(x._graph_value for x in variadic_args)
    sliding_inputs, global_inputs = kv_params.unflatten_basic_kv_tree(kv_inputs)
    kv_mapping = DeviceMapping.replicated(mesh)
    sliding_kv = PagedCacheValues.from_upstream(sliding_inputs, kv_mapping)
    global_kv = PagedCacheValues.from_upstream(global_inputs, kv_mapping)
    return sliding_kv, global_kv


__all__ = [
    "DiffusionGemmaDecoderTextModel",
    "DiffusionGemmaEncoderTextModel",
    "DiffusionGemmaSelfConditioning",
    "unflatten_kv",
]
