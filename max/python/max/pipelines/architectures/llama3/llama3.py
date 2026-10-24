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
"""Build a Llama3 model that uses continuous or paged kv-caching"""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.graph.quantization import QuantizationEncoding
from max.nn.attention import (
    AttentionWithRope,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
)
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParamInterface
from max.nn.layer import Module
from max.nn.linear import MLP, GPTQLinear, Linear
from max.nn.norm import ConstantLayerNorm, RMSNorm
from max.nn.transformer import DecoderOutputs, Transformer, TransformerBlock
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_kv_symbolic_input_tree,
)

from .model_config import Llama3Config, create_rope_embedding


class StackedMLP(Module):
    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear],
        has_scale: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length * 2,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        up_states = self.gate_up_proj(x)

        gate = up_states[:, : up_states.shape.static_dims[0] // 2]
        up_states = up_states[:, up_states.shape.static_dims[0] // 2 :]

        return self.down_proj(ops.silu(gate) * up_states)


class Llama3(Transformer):
    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) == 1
        self.config = config
        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=config.interleaved_rope_weights,
            rope_scaling_params=config.rope_scaling_params,
            longrope_scaling_params=config.longrope_scaling_params,
            device=config.devices[0],
        )

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.hidden_size,
                config.norm_dtype or config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,  # disable Gemma3-style scaling
            )
        else:
            create_norm = functools.partial(
                ConstantLayerNorm,
                config.hidden_size,
                config.devices[0],
                config.norm_dtype or config.dtype,
            )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinear, quantization_config=config.quantization_config
            )
        else:
            linear_cls = functools.partial(
                Linear, quant_config=config.quant_config
            )
        if config.stacked_mlp and config.quant_config:
            raise ValueError(
                "StackedMLP and scaled quantization are not compatible"
            )
        mlp_cls = (
            StackedMLP
            if config.stacked_mlp
            else functools.partial(MLP, quant_config=config.quant_config)
        )
        attention_cls: Callable[..., AttentionWithRope]
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            assert config.quantization_config is not None
            assert not config.attention_bias, (
                "Attention bias is not supported for GPTQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        elif config.model_quantization_encoding is not None:
            assert not config.attention_bias, (
                "Attention bias is not supported for GGUFQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GGUFQAttentionWithRope,
                quantization_encoding=config.model_quantization_encoding,
                scale=config.attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                AttentionWithRope,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
                has_bias=config.attention_bias,
                quant_config=config.quant_config,
            )

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
            embedding_multiplier=config.embedding_multiplier,
            logits_scaling=config.logits_scaling,
            target_layer_ids=config.target_layer_ids,
        )

    def input_tree(
        self,
        kv_params: KVCacheParamInterface,
        needs_hidden_state_input: bool = False,
    ) -> OrderedDict[str, object]:
        """Names the graph's inputs; its leaves are the graph's inputs."""
        # Hidden state input is for EAGLE-like spec decoding draft models.
        return ragged_kv_symbolic_input_tree(
            kv_params=kv_params,
            device_refs=list(self.config.devices),
            include_signal_buffers=False,
            hidden_states=(
                (self.config.dtype, self.config.hidden_size)
                if needs_hidden_state_input
                else None
            ),
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
        needs_hidden_state_input: bool = False,
    ) -> tuple[TensorType | BufferType, ...]:
        return tuple(
            tree.leaves(self.input_tree(kv_params, needs_hidden_state_input))
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

        Adapts the flat graph inputs to :meth:`__call__`: unflattens the KV
        cache inputs into the device's KV collection.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree` without a hidden state input.
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

        def forward(
            tokens: Value[Any],
            input_row_offsets: Value[Any],
            return_n_logits: Value[Any],
            *kv_inputs: Value[Any],
        ) -> DecoderOutputs:
            kv_collections = tree.leaves(
                self.kv_params.unflatten_kv_inputs(iter(kv_inputs)),
                leaf=KVCacheInputsPerDevice,
            )
            return self(
                tokens.tensor,
                kv_collections[0],
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
