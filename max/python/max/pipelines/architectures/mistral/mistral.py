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
"""Implements the Mistral model."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from max import tree
from max.graph import Graph, Type, Value
from max.graph import Module as GraphModule
from max.nn.attention import AttentionWithRope
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import DecoderOutputs, Transformer, TransformerBlock
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_kv_symbolic_input_tree,
)

from .model_config import MistralConfig


class Mistral(Transformer):
    """Defines the Mistral transformer model."""

    def __init__(self, config: MistralConfig) -> None:
        assert len(config.devices) == 1

        # hidden_size (5120) != num_attention_heads * head_dim (128 * 40 = 4,096)
        rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=False,
        )

        layers = [
            TransformerBlock(
                attention=AttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    devices=config.devices,
                    scale=config.attention_multiplier,
                    stacked_qkv=False,
                    has_bias=False,
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.feed_forward_length,
                    devices=config.devices,
                ),
                attention_norm=RMSNorm(
                    config.hidden_size,
                    config.dtype,
                    config.rms_norm_eps,
                    multiply_before_cast=False,
                ),
                mlp_norm=RMSNorm(
                    config.hidden_size,
                    config.dtype,
                    config.rms_norm_eps,
                    multiply_before_cast=False,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
        )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNorm(
                config.hidden_size,
                config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
        )
        self.devices = config.devices

    def input_tree(self) -> OrderedDict[str, object]:
        """Names the graph's inputs; its leaves are its inputs."""
        return ragged_kv_symbolic_input_tree(
            kv_params=self.kv_params,
            device_refs=list(self.devices),
            include_signal_buffers=False,
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

        Adapts the flat graph inputs to :meth:`__call__`, which takes the
        device's KV cache as one collection.

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
