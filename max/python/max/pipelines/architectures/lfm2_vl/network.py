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
"""Lfm2VlForConditionalGeneration's decoder on the ModuleV3 API.

The decoder is the LFM2 text model — 22 gated short-conv layers and 8
full-attention layers (QK-norm GQA, non-interleaved rope, theta 1e6),
dense SwiGLU MLP per layer, final `embedding_norm` (the name is
historical), tied lm_head — with one insertion: the image rows the vision
graph produced are scattered into the token embeddings at the
`image_token_id` (124907) placeholder positions before layer 0.

HF reference: modeling_lfm2_vl.py::Lfm2VlModel.forward —

    inputs_embeds = get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask,
                                                 image_features)
    outputs = language_model(inputs_embeds=...)

`masked_scatter` over a boolean token mask is `merge_multimodal_embeddings`
over the index list the pipeline's vision cache derives from each
image's token span, and the graph does one scatter.

The conv windows live in an engine-managed recurrent-state pool
mutated in place by causal_conv1d_varlen_fwd, and the attention KV in a
paged cache indexed by attention ordinal. NO `positions` input exists:
causal_conv1d_varlen_fwd resumes from the pool, not from an absolute
position ramp.
"""

from __future__ import annotations

import math
from typing import Any

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.nn import Module, as_subgraph
from max.nn.common_layers.attention import AttentionWithRope
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.embedding import Embedding
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    MHAKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.outputs import ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor

from .layers.conv import Lfm2ShortConv
from .layers.transformer_block import Lfm2VlTransformerBlock
from .model_config import ATTN_CACHE_KEY, STATE_CACHE_KEY, Lfm2VlConfig


def _attn_kv_params(kv_params: KVCacheParamInterface) -> Any:
    """The attention leaf of the cache tree.

    The offline weight dry-run builds the config with a bare
    head_dim-only stand-in for kv_params (check_weights_offline.
    build_config_kwargs); that has no ``children``/``child`` surface, and
    the attention layer only reads ``head_dim`` from it — pass it
    through, and resolve the stored tree's "attn" child when there is
    one (a flat KVCacheParams and a tree both reach this code).
    """
    children = getattr(kv_params, "children", None)
    if isinstance(children, dict) and children:
        if ATTN_CACHE_KEY in children:
            return children[ATTN_CACHE_KEY]
        return next(iter(children.values()))
    if hasattr(kv_params, "child"):
        return kv_params.child(ATTN_CACHE_KEY, MHAKVCacheParams)
    return kv_params


class Lfm2VlTextModel(
    Module[
        [
            Tensor,
            PagedCacheValues,
            RecurrentStateInputsPerDevice[Tensor, Tensor],
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ],
        ModuleV3Outputs,
    ]
):
    """The LFM2 hybrid decoder, with the image rows merged before layer 0."""

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.devices = config.devices
        self.config = config
        head_dim = config.head_dim or (
            config.hidden_size // config.num_attention_heads
        )
        if head_dim != config.hidden_size // config.num_attention_heads:
            raise ValueError(
                "the port assumes head_dim == hidden_size // "
                f"num_attention_heads ({config.hidden_size} // "
                f"{config.num_attention_heads} != {head_dim})"
            )
        rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=head_dim,
            interleaved=False,  # HF uses rotate_half, not interleaved
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        # HF's `model.language_model.embedding_norm` is the FINAL norm
        # (applied after the layer stack, before the head — the name is
        # historical; the monorepo lfm2 adapter maps it the same way).
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )
        attn_kv = _attn_kv_params(config.kv_params)
        layers = []
        attn_idx = 0
        for layer_type in config.layer_types:
            operator: Module[..., Tensor]
            if layer_type == "full_attention":
                # QK-norm GQA: q_layernorm/k_layernorm ([64], the
                # checkpoint says) rename onto the use_qk_norm leaves.
                operator = AttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=attn_kv,
                    layer_idx=attn_idx,
                    scale=math.sqrt(1.0 / head_dim),
                    has_bias=False,
                    use_qk_norm=True,
                    rms_norm_eps=config.norm_eps,
                )
                attn_idx += 1
            else:
                operator = Lfm2ShortConv(
                    hidden_size=config.hidden_size,
                    conv_L_cache=config.conv_L_cache,
                    dtype=config.dtype,
                )
            layers.append(
                Lfm2VlTransformerBlock(
                    layer_type=layer_type,
                    operator=operator,
                    feed_forward=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                        bias=False,
                    ),
                    operator_norm=RMSNorm(
                        config.hidden_size, eps=config.norm_eps
                    ),
                    ffn_norm=RMSNorm(config.hidden_size, eps=config.norm_eps),
                )
            )
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        # Logits TRANSPOSED: weight [vocab, hidden] @ h.T -> [vocab, M],
        # then transpose back. The h @ W.T orientation takes the thin gemv
        # path, which raises CUDA_ERROR_INVALID_VALUE on GB10 when fused
        # with the final-norm reduction at fp32 with N=vocab_size=128000
        # (the exaone phase-4 crash); the transposed form selects the GEMM.
        w = (
            self.embed_tokens.weight
            if self.tie_word_embeddings
            else self.lm_head.weight
        )
        logits_t = w @ F.transpose(h, 1, 0)
        return F.cast(F.transpose(logits_t, 1, 0), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        state: RecurrentStateInputsPerDevice[Tensor, Tensor],
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> ModuleV3Outputs:
        """Decode a ragged batch, merging image rows before layer 0."""
        conv = state.leaves[0]
        conv_pool = conv.pool
        conv_rows = conv.live_row_ids

        # A fresh prefill must start from a ZERO conv window, not from
        # whatever the pool row holds. The cache length counts tokens
        # already processed, so zero means this step's rows are fresh
        # and anything nonzero means the pool holds a prior window for
        # them (a comparison against the per-row token count would
        # misread a one-token prefill's first decode as fresh, and the
        # kernel would drop its conv window).
        has_initial_state = kv_collection.cache_lengths > 0
        # The kernel wants int32 cumulative row ends.
        query_start_loc = F.cast(input_row_offsets, DType.int32)

        h = self.embed_tokens(tokens)
        # The image rows replace the placeholder rows (an empty index
        # list is a no-op: the batch processor sends one for a
        # text-only step).
        h = F_merge_multimodal_embeddings(
            h, image_embeddings, image_token_indices
        ).rebind_mapping(h.mapping)

        conv_idx = 0
        for i, layer in enumerate(self.layers):
            conv_layer = F.constant(conv_idx, DType.int64, device=CPU())
            # The conv layers share one subgraph (the nemotron-h pool
            # precedent): the index rides in as a runtime input so the
            # shared graph selects each call's pool rows in-graph
            # (layers/conv.py). Attention layers cannot share one:
            # AttentionWithRope bakes its KV layer index in as a graph
            # constant, so a shared body would alias every attention
            # layer onto KV slot 0 (nemotron_h.py makes the same call).
            if self.config.layer_types[i] == "conv":
                conv_idx += 1
                layer_call = as_subgraph(layer, name="lfm2_conv_layer")
            else:
                layer_call = layer
            h = layer_call(
                h,
                kv_collection,
                conv_pool,
                conv_rows,
                conv_layer,
                query_start_loc,
                has_initial_state,
                input_row_offsets,
            )

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None
        # VARIABLE/ALL logits modes feed /v1/completions logprobs (the
        # teacher-forced parity walk and the eval gate both need them).
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
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
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
        return ModuleV3Outputs(
            next_token_logits=last_logits,
            logits=logits,
            logit_offsets=offsets,
            hidden_states=None,
        )


class Lfm2VlDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """The registered DECODER: unflatten the KV tree, run the text model.

    Registered under ``GraphRole.DECODER`` (see ``arch.py``); the
    framework's registered pipeline model constructs it from the
    finalized config, wraps it in the named-input graph and calls it
    with a ``DecoderInputs``. ``consumes_vision`` adds the ``vision``
    merge group, whose image rows the pipeline's vision cache encodes
    with the registered tower. ``input_layout`` pins the operand order
    the graph had before the registration (tokens, logits mask, row
    offsets, vision merge, KV) — the registration names the same inputs
    as a tree, so the order only shows up in the exported graph.
    """

    graph_name = "Lfm2VlForConditionalGeneration"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.language_model = Lfm2VlTextModel(config)
        self.kv_params = config.kv_params

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Unflatten the KV tree and run the decoder.

        ``unflatten_kv_inputs`` rebuilds the same tree
        ``construct_kv_params`` declared. The KV leaves arrive as Tensors
        in every mode — staged, compiled and eager — and
        ``from_upstream`` takes tensors, so they pass through unchanged.
        """
        tokens = inputs.tokens
        vision = inputs.vision
        assert vision is not None  # consumes_vision: the graph always takes it
        kv_tree: Any = self.kv_params.unflatten_kv_inputs(
            iter(tree.leaves(inputs.kv))
        )
        attn_leaves = tree.leaves(
            kv_tree[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice
        )
        state_leaves = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        kv_collection = PagedCacheValues.from_upstream(
            attn_leaves, tokens.mapping
        )
        return self.language_model(
            tokens,
            kv_collection,
            state_leaves[0],
            inputs.return_n_logits,
            inputs.input_row_offsets,
            vision.embeddings,
            vision.scatter_indices,
        )
