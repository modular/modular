# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Build a Qwen2.5VL decoder model that uses paged kv-caching.

This module implements the Qwen2.5VL decoder architecture with support for:
- 3D position IDs and multi-axis rotary position embedding (mrope)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn import (
    MLP,
    Embedding,
    LayerList,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    ReturnLogits,
    RMSNorm,
)
from max.nn.kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheCollection,
)
from max.nn.layer import Layer
from max.nn.rotary_embedding import RotaryEmbedding
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)
from max.pipelines.architectures.llama3.llama3 import StackedMLP
from max.pipelines.architectures.qwen2_5vl.model_config import Qwen2_5VLConfig


class Qwen25VLDecoderAttentionWithRope(Module):
    """Qwen2.5VL attention layer with multi-axis rotary position embedding (mrope).

    This attention implementation supports 2D position IDs for vision-language tasks.
    """

    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = True,
    ) -> None:
        """Initializes the Qwen2.5VL attention layer with mrope support.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            dtype: DType of the QKV and output projection weights.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        self.devices = devices or [DeviceRef.CPU()]

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
        )

        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""

        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight

        wqkv = ops.concat((wq, wk, wv))
        return wqkv

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        # Access bias, which should all exist since has_bias=True.
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedKVCacheCollection,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        position_ids: TensorValue,
        mrope_section: list[int],
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        freqs_cis = freqs_cis.to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
            position_ids=position_ids,
            mrope_section=mrope_section,
        )
        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)


class Qwen25VLDecoderTransformerBlock(Module):
    """Qwen2.5VL decoder transformer block customized for supporting 2D position ids."""

    def __init__(
        self,
        attention: Qwen25VLDecoderAttentionWithRope,
        mlp: Layer,
        attention_norm: Layer,
        mlp_norm: Layer,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp

        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedKVCacheCollection,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        position_ids: TensorValue,
        mrope_section: list[int],
    ) -> TensorValue:
        attn_out = self.self_attn(
            layer_idx,
            self.input_layernorm(x),
            kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
            position_ids=position_ids,
            mrope_section=mrope_section,
        )

        h = x + attn_out
        mlp = self.mlp(self.post_attention_layernorm(h))

        return h + mlp


class Qwen25VLDecoder(Module):
    """Qwen2.5VL decoder model with support for vision-language tasks.

    This decoder implements the Qwen2.5VL architecture with:
    - Multi-axis rotary position embeddings (mrope) for 2D position encoding
    """

    def __init__(self, config: Qwen2_5VLConfig) -> None:
        assert len(config.devices) == 1
        rope = Llama3RotaryEmbedding(
            dim=config.llm_config.hidden_size,
            n_heads=config.llm_config.num_attention_heads,
            theta=config.llm_config.rope_theta,
            max_seq_len=config.llm_config.max_seq_len,
            head_dim=config.llm_config.kv_params.head_dim,
            interleaved=config.llm_config.interleaved_rope_weights,
            scaling_params=config.llm_config.rope_scaling_params,
            device=config.devices[0],
        )

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.llm_config.norm_method == "rms_norm":
            if config.llm_config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.llm_config.hidden_size,
                config.llm_config.norm_dtype or config.llm_config.dtype,
                config.llm_config.rms_norm_eps,
                multiply_before_cast=False,  # disable Gemma3-style scaling
            )
        else:
            raise ValueError(
                f"Unsupported norm method: {config.llm_config.norm_method}"
            )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]

        linear_cls = functools.partial(Linear)

        mlp_cls = (
            StackedMLP
            if config.llm_config.stacked_mlp
            else functools.partial(MLP)
        )
        attention_cls: Callable[..., Qwen25VLDecoderAttentionWithRope]

        attention_cls = functools.partial(
            Qwen25VLDecoderAttentionWithRope,
            scale=config.llm_config.attention_multiplier,
            has_bias=config.llm_config.attention_bias,
        )

        layers = [
            Qwen25VLDecoderTransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.llm_config.num_attention_heads,
                    num_key_value_heads=config.llm_config.num_key_value_heads,
                    hidden_size=config.llm_config.hidden_size,
                    kv_params=config.llm_config.kv_params,
                    dtype=config.llm_config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.llm_config.dtype,
                    config.llm_config.model_quantization_encoding,
                    config.llm_config.hidden_size,
                    config.llm_config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
            )
            for i in range(config.llm_config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.llm_config.dtype

        embedding_layer = Embedding(
            config.llm_config.vocab_size,
            config.llm_config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output = Linear(
            config.llm_config.hidden_size,
            config.llm_config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
        )

        if config.llm_config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        kv_collection_cls: type[FetchPagedKVCacheCollection]

        if config.llm_config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy "
                + str(config.llm_config.kv_params.cache_strategy)
            )

        super().__init__()
        self.dim = config.llm_config.hidden_size
        self.n_heads = config.llm_config.num_attention_heads
        self.layers = LayerList(layers)
        self.norm = create_norm()
        self.lm_head = output
        self.embed_tokens = embedding_layer
        self.kv_params = config.llm_config.kv_params
        self.kv_collection_constructor = kv_collection_cls(
            config.llm_config.kv_params,
            num_layers=config.llm_config.num_hidden_layers,
        )
        self.logits_postprocessor = config.llm_config.logits_postprocessor
        self.rope = rope
        self.return_logits = config.llm_config.return_logits

    def _apply_logits_postprocessor(
        self, output: tuple[TensorValue, ...]
    ) -> tuple[TensorValue, ...]:
        if self.logits_postprocessor is None:
            return output
        return tuple(self.logits_postprocessor(elem) for elem in output)

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        image_embeddings: TensorValue,
        image_token_indices: TensorValue,
        position_ids: TensorValue,
        mrope_section: list[int],
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens)

        h = merge_multimodal_embeddings(
            inputs_embeds=h,
            multimodal_embeddings=image_embeddings,
            image_token_indices=image_token_indices,
        )

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        # Create position embeddings shared across the decoder layers.
        freqs_cis = self.rope.freqs_cis

        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
                position_ids=position_ids,
                mrope_section=mrope_section,
            )

        # Retrieve a variable number of tokens
        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = ops.cast(self.lm_head(self.norm(last_h)), DType.float32)
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
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            last_tokens = ops.gather(h, last_indices, axis=0)
            logits = ops.cast(
                self.lm_head(self.norm(last_tokens)), DType.float32
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h)), DType.float32)
            offsets = input_row_offsets

        if logits:
            last_logits, logits = self._apply_logits_postprocessor(
                (
                    last_logits,
                    logits,
                )
            )
        else:
            last_logits = self._apply_logits_postprocessor((last_logits,))[0]

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
