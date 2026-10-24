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
"""Build a Llama3 model that runs on multiple devices."""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    Type,
    Value,
)
from max.graph import Module as GraphModule
from max.graph.quantization import QuantizationEncoding
from max.nn.attention import TensorParallelAttentionWithRope
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParamInterface
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.norm import RMSNorm
from max.nn.transformer import (
    DecoderOutputs,
    DistributedTransformer,
    DistributedTransformerBlock,
)
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_kv_symbolic_input_tree,
)

logger = logging.getLogger("max.pipelines")
from .model_config import Llama3Config, create_rope_embedding


class DistributedLlama3(DistributedTransformer):
    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) > 1
        self.config = config

        if config.quantization_config:
            raise ValueError(
                "Model contains GPTQ weights. This is currently not supported with multiple GPUs."
            )

        if config.stacked_mlp:
            raise ValueError(
                "Model contains stacked MLP weights. This is currently not supported with multiple GPUs."
            )

        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "`norm_method` must be `RMSNorm` and `rms_norm_eps` cannot be "
                "None for model that uses `RMSNorm`."
            )

        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=config.interleaved_rope_weights,
            rope_scaling_params=config.rope_scaling_params,
            longrope_scaling_params=config.longrope_scaling_params,
            device=DeviceRef.CPU(),
        )

        create_distributed_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            dtype=config.norm_dtype or config.dtype,
            eps=config.rms_norm_eps,
        )

        quant_cfg = config.quant_config
        linear_cls = functools.partial(Linear, quant_config=quant_cfg)

        layers = []
        sublayer_groupings_dict = defaultdict(list)

        for layer_idx in range(config.num_hidden_layers):
            # Deal with the float8 case where individual layers are ignored
            # specially: assume bfloat16 dtype for "ignored" layers in fp8
            # quantized models.
            attn_qkv_dtype = (
                DType.bfloat16
                if quant_cfg
                and layer_idx not in quant_cfg.attn_quantized_layers
                else config.dtype
            )
            mlp_dtype = (
                DType.bfloat16
                if quant_cfg and layer_idx not in quant_cfg.mlp_quantized_layers
                else config.dtype
            )

            sublayer_groupings_dict[(attn_qkv_dtype, mlp_dtype)].append(
                layer_idx
            )

            mlp = MLP(
                mlp_dtype,
                config.model_quantization_encoding,
                config.hidden_size,
                config.intermediate_size,
                config.devices,
                linear_cls,
                quant_config=(
                    quant_cfg
                    if quant_cfg
                    and (layer_idx in quant_cfg.mlp_quantized_layers)
                    else None
                ),
            )

            layers.append(
                DistributedTransformerBlock(
                    attention=TensorParallelAttentionWithRope(
                        stacked_qkv=config.stacked_qkv,
                        scale=config.attention_multiplier,
                        clip_qkv=config.clip_qkv,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        dtype=attn_qkv_dtype,
                        rope=rope,
                        linear_cls=linear_cls,
                        devices=config.devices,
                        has_bias=config.attention_bias,
                        # Only pass the float8 config if this attention layer is quantized.
                        quant_config=(
                            quant_cfg
                            if quant_cfg
                            and (layer_idx in quant_cfg.attn_quantized_layers)
                            else None
                        ),
                    ),
                    mlp=mlp,
                    attention_norm=create_distributed_norm(),
                    mlp_norm=create_distributed_norm(),
                    devices=config.devices,
                    # TODO: Support residual_multiplier
                    # residual_multiplier=config.residual_multiplier,
                )
            )

        subgraph_layer_groups = list(sublayer_groupings_dict.values())

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if quant_cfg and quant_cfg.embedding_output_dtype:
            embedding_output_dtype = quant_cfg.embedding_output_dtype

        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=embedding_output_quantization,
        )
        output = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            tied_weight=(
                embedding_layer.weight if config.tie_word_embeddings else None
            ),
            quantization_encoding=embedding_output_quantization,
        )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_distributed_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            devices=config.devices,
            rope=rope,
            return_logits=config.return_logits,
            use_subgraphs=config.use_subgraphs,
            subgraph_layer_groups=subgraph_layer_groups,
            # TODO: Support the following config options.
            # embedding_multiplier=config.embedding_multiplier,
            logits_scaling=config.logits_scaling,
        )

    def input_tree(
        self, kv_params: KVCacheParamInterface
    ) -> OrderedDict[str, object]:
        """Names the graph's inputs; its leaves are the graph's inputs."""
        return ragged_kv_symbolic_input_tree(
            kv_params=kv_params,
            device_refs=list(self.config.devices),
            include_signal_buffers=True,
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        return tuple(tree.leaves(self.input_tree(kv_params)))

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_tree`'s leaves.

        Adapts the flat graph inputs to :meth:`__call__`: splits off a
        signal buffer per device and unflattens the KV cache inputs into
        each device's KV collection.

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
        n_devices = len(self.devices)

        def forward(
            tokens: Value[Any],
            input_row_offsets: Value[Any],
            return_n_logits: Value[Any],
            *variadic_args: Value[Any],
        ) -> DecoderOutputs:
            signal_buffers = [v.buffer for v in variadic_args[:n_devices]]
            kv_collections = tree.leaves(
                self.kv_params.unflatten_kv_inputs(
                    iter(variadic_args[n_devices:])
                ),
                leaf=KVCacheInputsPerDevice,
            )
            return self(
                tokens.tensor,
                signal_buffers,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
