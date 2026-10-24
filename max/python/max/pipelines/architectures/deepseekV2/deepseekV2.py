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
"""Implements the DeepseekV2 model."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
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
from max.nn.attention.multi_latent_attention import (
    LatentAttentionWithRope,
)
from max.nn.comm import Signals
from max.nn.embedding import Embedding
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.linear import MLP, Linear
from max.nn.moe import MoE
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.transformer import DecoderOutputs, Transformer, TransformerBlock
from max.pipelines.architectures.deepseekV2.layers.moe_gate import (
    DeepSeekV2MoEGate,
)

from .model_config import DeepseekV2Config


def deepseekV2_input_types(
    devices: Sequence[DeviceRef], kv_params: KVCacheParamInterface
) -> tuple[TensorType | BufferType, ...]:
    """Returns the DeepseekV2 graph's input types.

    The tokens, row offsets and ``return_n_logits``, all on the first device,
    then one signal buffer per device when there are several, then the KV
    cache.

    Args:
        devices: The devices the graph runs on.
        kv_params: The KV cache parameters.
    """
    device_ref = devices[0]
    signals = Signals(devices=devices).input_types() if len(devices) > 1 else []
    return (
        TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
        TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        ),
        TensorType(DType.int64, shape=["return_n_logits"], device=device_ref),
        *signals,
        *kv_params.flattened_kv_inputs(),
    )


def unflatten_kv_collections(
    kv_params: KVCacheParamInterface, kv_inputs: Iterable[Value[Any]]
) -> list[PagedCacheValues]:
    """Returns each device's KV cache collection from the flat KV inputs."""
    return tree.leaves(
        kv_params.unflatten_kv_inputs(iter(kv_inputs)),
        leaf=KVCacheInputsPerDevice,
    )


class DeepseekV2(Transformer):
    """Defines the DeepseekV2 transformer model."""

    def __init__(self, config: DeepseekV2Config) -> None:
        assert len(config.devices) == 1
        assert config.rope_scaling is not None
        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        rope = DeepseekYarnRotaryEmbedding(
            config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
        )

        layers = [
            TransformerBlock(
                attention=LatentAttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    devices=config.devices,
                    graph_mode=config.graph_mode,
                    buffer_size=config.max_batch_context_length,
                ),
                mlp=self._get_mlp(config, i),
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
        lm_head = self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            device=config.devices[0],
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
            output=lm_head,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
        )
        self.devices = config.devices

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Returns the graph's input types; see
        :func:`deepseekV2_input_types`."""
        return deepseekV2_input_types(self.devices, self.kv_params)

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_types`.

        Adapts the flat graph inputs to :meth:`__call__` by unflattening the
        KV cache.

        Args:
            *input_types: The graph's input types, :meth:`input_types`.
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
            kv_collections = unflatten_kv_collections(self.kv_params, kv_inputs)
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

    def _get_mlp(self, config: DeepseekV2Config, i: int) -> Module:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            i: Layer index

        Returns:
            Either a MoE or MLP module depending on the layer index and config
        """
        if (
            config.n_routed_experts is not None
            and i >= config.first_k_dense_replace
            and i % config.moe_layer_freq == 0
        ):
            assert len(config.devices) == 1, "Expect only one device"
            return MoE(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.n_routed_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=functools.partial(
                    DeepSeekV2MoEGate,
                    topk_method=config.topk_method,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                    routed_scaling_factor=config.routed_scaling_factor,
                ),
                has_shared_experts=True,
                shared_experts_dim=config.n_shared_experts
                * config.moe_intermediate_size,
                dtype=config.dtype,
            )
        else:
            return MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
            )
