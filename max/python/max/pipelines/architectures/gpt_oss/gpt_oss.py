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

"""Implements the GPT OSS model."""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    Graph,
    ShardingStrategy,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import ColumnParallelLinear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.rotary_embedding import (
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from max.nn.transformer import DecoderOutputs
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_batch_inputs,
    signal_inputs,
)

from .layers.attention import GptOssAttention
from .layers.moe import GptOssMoE
from .layers.transformer_block import GptOssTransformerBlock
from .model_config import GptOssConfig


class GptOssTextModel(DistributedLogitsPostprocessMixin, Module):
    """The GPT OSS language model.

    Decoder-only Transformer with MoE feed-forward, rotary embeddings (YARN),
    and mixed attention (full + sliding window).
    """

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.devices = config.devices

        # Set embedding output dtype for quantized models
        self.embedding_output_dtype: DType | None = None
        if config.quant_config and config.quant_config.embedding_output_dtype:
            self.embedding_output_dtype = (
                config.quant_config.embedding_output_dtype
            )

        # Create YARN scaling params if configured
        assert config.rope_scaling is not None, (
            "RoPE scaling is required for GPT-OSS models"
        )
        assert isinstance(config.rope_scaling, YarnScalingParams), (
            "Only YARN scaling is supported for GPT-OSS models"
        )
        yarn_scaling_params: YarnScalingParams = config.rope_scaling

        # RoPE with YARN scaling for full and window attention layers
        rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.dtype,
            config.rms_norm_eps,
            multiply_before_cast=True,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=True,
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_layer_type: dict[str, KVCacheParams] = {}
        for layer_type_key, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_layer_type[layer_type_key] = kv_params_leaf

        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(config.num_hidden_layers):
            layer_type = (
                config.layer_types[i]
                if i < len(config.layer_types)
                else "full_attention"
            )
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                GptOssTransformerBlock(
                    attention=GptOssAttention(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params_by_layer_type[layer_type],
                        layer_idx=layer_idx_in_cache,
                        dtype=config.dtype,
                        devices=config.devices,
                        local_window_size=config.sliding_window,
                        has_bias=config.attention_bias,
                        layer_type=layer_type,
                    ),
                    mlp=GptOssMoE(config),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    devices=config.devices,
                )
            )

        self._layer_kv_key = [
            config.layer_types[i]
            if i < len(config.layer_types)
            else "full_attention"
            for i in range(config.num_hidden_layers)
        ]

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        **kwargs,
    ) -> DecoderOutputs:
        h_embed = self.embed_tokens(tokens)
        if self.embedding_output_dtype is not None:
            h_embed = ops.cast(h_embed, self.embedding_output_dtype)
        # Replicate embedding output to all devices
        h = [h_embed.to(device) for device in self.devices]

        kv_collections_by_type = {
            "sliding_attention": sliding_kv_collections,
            "full_attention": global_kv_collections,
        }

        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
                **kwargs,
            )

        return self._postprocess_logits(
            h, input_row_offsets, return_n_logits, signal_buffers
        )


class GptOss(Module):
    """The GPT OSS model."""

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.language_model = GptOssTextModel(config)

    def input_tree(self) -> OrderedDict[str, object]:
        """Names the graph's inputs; its leaves are its inputs.

        Returns:
            The input types under ``batch`` (whose ``row_offsets`` are one
            per device), ``signals`` and ``kv``, in graph order.
        """
        devices = list(self.language_model.devices)
        return OrderedDict(
            batch=ragged_batch_inputs(
                devices[0],
                ("tokens", "return_n_logits", "row_offsets"),
                row_offsets_devices=devices,
            ),
            signals=signal_inputs(devices),
            kv=self.language_model.kv_params.get_symbolic_inputs(),
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_tree`'s leaves.

        Adapts the flat graph inputs to :meth:`__call__`, splitting the KV
        cache into its sliding and global collections.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        n_devices = len(self.language_model.devices)

        def forward(
            tokens: Value[Any],
            return_n_logits: Value[Any],
            *variadic_args: Value[Any],
        ) -> DecoderOutputs:
            input_row_offsets = [v.tensor for v in variadic_args[:n_devices]]
            signal_buffers = [
                v.buffer for v in variadic_args[n_devices : 2 * n_devices]
            ]
            sliding_kv, global_kv = (
                self.language_model.kv_params.unflatten_basic_kv_tree(
                    iter(variadic_args[2 * n_devices :])
                )
            )
            return self(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                sliding_kv_collections=sliding_kv,
                global_kv_collections=global_kv,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
    ) -> DecoderOutputs:
        return self.language_model(
            tokens,
            signal_buffers,
            sliding_kv_collections,
            global_kv_collections,
            return_n_logits,
            input_row_offsets,
        )
