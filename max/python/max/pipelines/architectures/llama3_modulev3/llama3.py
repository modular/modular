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

"""Implements the Llama3 model using the ModuleV3 API."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ClassVar

from max import functional as F
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from max.nn.common_layers.attention import AttentionWithRope
from max.nn.common_layers.mlp import MLP
from max.nn.embedding import Embedding
from max.nn.legacy.data_parallelism import split_batch_replicated
from max.nn.legacy.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.linear import Linear
from max.nn.norm import LayerNorm, RMSNorm
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lora import LoRATargetModule
from max.tensor import Tensor

from .layers.mlp import LlamaStackedMLP
from .layers.rotary_embedding import (
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
)
from .layers.transformer_block import LlamaTransformerBlock
from .lora import LLAMA3_LORA_TARGETS
from .model_config import Llama3Config


class Llama3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], ModuleV3Outputs]
):
    """The Llama3 language model.

    Decoder-only Transformer with SwiGLU MLP, rotary embeddings,
    and grouped-query attention.
    """

    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.devices = config.devices

        # Create RoPE embedding.
        rope: Llama3RotaryEmbedding | LongRoPERotaryEmbedding
        if config.longrope_scaling_params is not None:
            rope = LongRoPERotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                device=config.devices[0].to_device(),
                head_dim=Llama3Config.get_head_dim_from_config(config),
                interleaved=config.interleaved_rope_weights,
                scaling_params=config.longrope_scaling_params,
            )
        else:
            rope = Llama3RotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                device=config.devices[0].to_device(),
                head_dim=Llama3Config.get_head_dim_from_config(config),
                interleaved=config.interleaved_rope_weights,
                scaling_params=config.rope_scaling_params,
            )

        # Select norm type.
        create_norm: Callable[..., Module[[Tensor], Tensor]]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm, config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            create_norm = functools.partial(
                LayerNorm,
                config.hidden_size,
                elementwise_affine=config.norm_elementwise_affine,
            )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = create_norm()

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        # Build transformer layers.
        layers = []
        for i in range(config.num_hidden_layers):
            mlp: MLP | LlamaStackedMLP
            if config.stacked_mlp:
                mlp = LlamaStackedMLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                )
            else:
                mlp = MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                )

            layers.append(
                LlamaTransformerBlock(
                    attention=AttentionWithRope(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        layer_idx=i,
                        scale=config.attention_multiplier,
                        has_bias=config.attention_bias,
                        stacked_qkv=config.stacked_qkv,
                        clip_qkv=config.clip_qkv,
                    ),
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Compute logits from hidden states, handling weight tying."""
        if self.tie_word_embeddings:
            return F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the text model over the graph inputs."""
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * float(self.embedding_multiplier)

        # Run through transformer layers.
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Compute logits based on return mode.
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        hidden_states = None
        if self.return_hidden_states == ReturnHiddenStates.LAST:
            hidden_states = last_h
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            hidden_states = self.norm(h)

        return ModuleV3Outputs(
            next_token_logits=last_logits,
            logits=logits,
            logit_offsets=offsets,
            hidden_states=hidden_states,
        )


class Llama3(Module[..., ModuleV3Outputs]):
    """The Llama3 model.

    Top-level wrapper that unflattens the variadic KV cache arguments
    and delegates to :class:`Llama3TextModel`.
    """

    def __init__(
        self,
        config: Llama3Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Llama3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> ModuleV3Outputs:
        """Unflattens the KV inputs, then runs the text model."""
        kv_inputs = iter(x._graph_value for x in variadic_args)
        symbolic_inputs = self.kv_params.unflatten_kv_inputs(kv_inputs)
        kv_collections = tree.leaves(
            symbolic_inputs, leaf=KVCacheInputsPerDevice
        )
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )


def _replica(module: Llama3TextModel, index: int) -> Llama3TextModel:
    """Returns ``module`` with each tensor replaced by its shard on one device.

    The module's weights are replicated over the data-parallel mesh, so the
    shard is a whole copy, and the replica runs on that device alone.
    """
    return tree.map(
        lambda tensor: tensor.local_shards[index],
        module,
        leaf=Tensor,
        shared=True,
    )


def _split_batch(
    devices: list[DeviceRef], inputs: DecoderInputs
) -> list[tuple[Tensor, Tensor]]:
    """Returns each replica's tokens and row offsets, on its device."""
    assert inputs.host_input_row_offsets is not None
    assert inputs.data_parallel_splits is not None
    tokens = TensorValue(inputs.tokens)
    input_row_offsets = TensorValue(inputs.input_row_offsets)
    split_tokens, split_offsets = split_batch_replicated(
        devices,
        [tokens.to(device) for device in devices],
        [input_row_offsets.to(device) for device in devices],
        TensorValue(inputs.host_input_row_offsets).cast(DType.int64),
        TensorValue(inputs.data_parallel_splits),
    )
    return [
        (
            Tensor.from_graph_value(replica_tokens),
            Tensor.from_graph_value(offsets),
        )
        for replica_tokens, offsets in zip(
            split_tokens, split_offsets, strict=True
        )
    ]


class Llama3Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """The Llama 3 decoder, registered for the ``DECODER`` role.

    On several devices each device runs a whole replica of
    :class:`Llama3TextModel` over its own slice of the batch.

    Raises:
        ValueError: If it runs on several devices without one data-parallel
            replica per device.
        NotImplementedError: If it runs data-parallel and must return more
            than the last token's logits.
    """

    graph_name: ClassVar[str] = "Llama3"
    lora_targets: ClassVar[tuple[LoRATargetModule, ...]] = LLAMA3_LORA_TARGETS
    supports_data_parallel: ClassVar[bool] = True

    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        num_replicas = config.kv_params.data_parallel_degree
        if num_replicas != len(config.devices):
            raise ValueError(
                "Llama 3 on ModuleV3 has no tensor parallelism: run one "
                f"data-parallel replica per device, not {num_replicas} over "
                f"{len(config.devices)} devices"
            )
        if num_replicas > 1 and (
            config.return_logits != ReturnLogits.LAST_TOKEN
            or config.return_hidden_states != ReturnHiddenStates.NONE
        ):
            raise NotImplementedError(
                "data-parallel Llama 3 returns only the last token's logits"
            )
        self.language_model = Llama3TextModel(config)
        self.kv_params = config.kv_params
        self.devices = config.devices

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs each replica over its sequences, then joins their logits."""
        kv_inputs = iter(x._graph_value for x in tree.leaves(inputs.kv))
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(kv_inputs),
            leaf=KVCacheInputsPerDevice,
        )
        if inputs.data_parallel_splits is None:
            return self.language_model(
                inputs.tokens,
                kv_collections[0],
                inputs.return_n_logits,
                inputs.input_row_offsets,
            )
        first_device = self.devices[0].to_device()
        logits = []
        for index, (tokens, input_row_offsets) in enumerate(
            _split_batch(self.devices, inputs)
        ):
            outputs = _replica(self.language_model, index)(
                tokens,
                kv_collections[index],
                inputs.return_n_logits,
                input_row_offsets,
            )
            logits.append(outputs.next_token_logits.to(first_device))
        return DecoderOutputs(next_token_logits=F.concat(logits, axis=0))
