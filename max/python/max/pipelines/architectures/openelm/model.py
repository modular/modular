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
"""MAX PipelineModel implementation for Apple's OpenELM architecture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max import tree
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.attention import AttentionWithRope
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.layer.layer_list import LayerList
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import ReturnLogits
from max.nn.transformer.transformer import LogitsPostprocessMixin
from max.pipelines.architectures.llama3.model import LlamaModelBase

from .model_config import OpenELMConfig, OpenELMLayerConfig


class OpenELMFeedForward(Module):
    """OpenELM's SwiGLU FFN: a single fused gate+up projection, then down."""

    def __init__(
        self,
        *,
        hidden_size: int,
        ffn_hidden_dim: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.ffn_hidden_dim = ffn_hidden_dim
        self.proj_1 = Linear(
            in_dim=hidden_size,
            out_dim=ffn_hidden_dim * 2,
            dtype=dtype,
            device=device,
        )
        self.proj_2 = Linear(
            in_dim=ffn_hidden_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        y12 = self.proj_1(x)
        y1 = y12[:, : self.ffn_hidden_dim]
        y2 = y12[:, self.ffn_hidden_dim :]
        return self.proj_2(ops.silu(y1) * y2)


class OpenELMTransformerBlock(Module):
    """One OpenELM layer: GQA attention (this layer's real head counts) + SwiGLU FFN."""

    def __init__(
        self,
        *,
        layer_cfg: OpenELMLayerConfig,
        kv_params: KVCacheParams,
        hidden_size: int,
        dtype: DType,
        rms_norm_eps: float,
        normalize_qk_projections: bool,
        device: DeviceRef,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, dtype, eps=rms_norm_eps)
        self.attn = AttentionWithRope(
            rope=rope,
            num_attention_heads=layer_cfg.num_query_heads,
            num_key_value_heads=layer_cfg.num_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=[device],
            dtype=dtype,
            stacked_qkv=True,
            use_qk_norm=normalize_qk_projections,
            rms_norm_eps=rms_norm_eps,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
        )
        self.ffn_norm = RMSNorm(hidden_size, dtype, eps=rms_norm_eps)
        self.ffn = OpenELMFeedForward(
            hidden_size=hidden_size,
            ffn_hidden_dim=layer_cfg.ffn_hidden_dim,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        attn_out = self.attn(
            layer_idx,
            self.attn_norm(x),
            kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
        )
        h = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(h))
        return h + ffn_out


class OpenELMBody(Module):
    """Everything the checkpoint stores under the ``transformer.`` prefix."""

    def __init__(
        self,
        config: OpenELMConfig,
        device: DeviceRef,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(
            config.vocab_size, config.model_dim, config.dtype, device
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_group: dict[str, KVCacheParams] = {}
        for group_key, leaf in config.kv_params.children.items():
            assert isinstance(leaf, KVCacheParams)
            kv_params_by_group[group_key] = leaf

        layers = []
        self._layer_group_key: list[str] = []
        local_idx_by_group: dict[str, int] = {}
        self._layer_local_idx: list[int] = []
        for layer_cfg in config.layer_configs:
            group_key = f"kv_heads_{layer_cfg.num_kv_heads}"
            local_idx = local_idx_by_group.get(group_key, 0)
            self._layer_group_key.append(group_key)
            self._layer_local_idx.append(local_idx)
            local_idx_by_group[group_key] = local_idx + 1
            layers.append(
                OpenELMTransformerBlock(
                    layer_cfg=layer_cfg,
                    kv_params=kv_params_by_group[group_key],
                    hidden_size=config.model_dim,
                    dtype=config.dtype,
                    rms_norm_eps=config.rms_norm_eps,
                    normalize_qk_projections=config.normalize_qk_projections,
                    device=device,
                    rope=rope,
                )
            )
        self.layers = LayerList(layers)
        self._group_keys_in_order = list(kv_params_by_group.keys())
        self._rope = rope

    def __call__(
        self,
        tokens: TensorValue,
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        kv_collections_by_group = dict(
            zip(self._group_keys_in_order, kv_collections, strict=True)
        )
        h = self.token_embeddings(tokens)
        freqs_cis = self._rope.freqs_cis
        device = h.device

        for idx, layer in enumerate(self.layers):
            group_key = self._layer_group_key[idx]
            layer_idx = ops.constant(
                self._layer_local_idx[idx], DType.uint32, device=device
            )
            h = layer(
                layer_idx,
                h,
                kv_collections_by_group[group_key],
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        return h


class OpenELMLanguageModel(LogitsPostprocessMixin, Module):
    """OpenELM's graph: per-layer KV cache groups, weight-tied output head."""

    def __init__(
        self, config: OpenELMConfig, *, return_logits: ReturnLogits
    ) -> None:
        super().__init__()
        device = config.devices[0]
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            # OpenELM's head COUNT varies per layer, but head_dim (and so the
            # RoPE angle table) is constant across layers; n_heads is unused
            # here since head_dim is given explicitly.
            n_heads=1,
            theta=config.rope_base,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            interleaved=True,
        )
        self.transformer = OpenELMBody(config, device, self.rope)

        self.lm_head = Linear(
            in_dim=config.model_dim,
            out_dim=config.vocab_size,
            dtype=config.dtype,
            device=device,
        )
        self.lm_head.set_shared_weight(
            "weight", self.transformer.token_embeddings.weight
        )

        # Kept top-level (not nested under `transformer`): the weight
        # adapter renames the checkpoint's "transformer.norm.weight" key
        # to "norm.weight" to match.
        self.norm = RMSNorm(
            config.model_dim, config.dtype, eps=config.rms_norm_eps
        )
        self.return_logits = return_logits

    def __call__(
        self,
        tokens: TensorValue,
        kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        h = self.transformer(tokens, kv_collections, input_row_offsets)
        return self._postprocess_logits(h, input_row_offsets, return_n_logits)


class OpenELMModel(LlamaModelBase):
    """MAX PipelineModel for Apple's OpenELM family (270M / 450M / 1.1B / 3B)."""

    model_config_cls = OpenELMConfig

    def _create_model_config(self, state_dict: dict[str, Any]) -> OpenELMConfig:
        del state_dict
        return self.arch_config_as(OpenELMConfig)

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: OpenELMConfig,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        device_ref = self.device_refs[0]

        # Input order must match the inherited execute()'s expected buffer
        # order: tokens, input row offsets, return_n_logits, then KV cache.
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        nn_model = OpenELMLanguageModel(
            model_config, return_logits=self.return_logits
        )
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        weights_registry = nn_model.state_dict(auto_initialize=False)

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = tree.leaves(kv_inputs)

        with Graph(
            "OpenELMForCausalLM",
            input_types=[
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *flattened_kv_types,
            ],
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_args = graph.inputs

            assert isinstance(self.kv_params, MultiKVCacheParams)
            kv_collections = self.kv_params.unflatten_basic_kv_tree(
                iter(kv_args)
            )
            # unflatten_basic_kv_tree returns one entry per attention child;
            # each entry is a per-device list, so take the single device's.
            kv_collections_single_device = [
                per_device[0] for per_device in kv_collections
            ]

            outputs = nn_model(
                tokens=tokens.tensor,
                kv_collections=kv_collections_single_device,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
            )
            graph.output(*outputs)
        return graph, weights_registry
