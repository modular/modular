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
from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn import (
    MLP,
    ColumnParallelLinear,
    LayerList,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    ReturnLogits,
    RMSNorm,
    VocabParallelEmbedding,
)
from max.nn.attention.attention_with_rope import _compute_shard_range
from max.nn.comm.allreduce import Allreduce
from max.nn.kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Shardable
from max.nn.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings_with_gather,
)
from max.pipelines.architectures.internvl.internvl import distribute_value
from max.pipelines.architectures.llama3.model_config import Llama3Config


class Qwen25VLDecoderAttentionWithRope(Module, Shardable):
    """Qwen2.5VL attention layer with multi-axis rotary position embedding (mrope).

    This implementation is based on the Qwen2.5VL language model architecture, which
    is similar to Llama3 but includes attention bias and multi-axis rotary position embedding (mrope).

    This is a distributed attention layer that supports tensor parallel and replicate sharding strategies.

    This attention implementation supports 2D position IDs for vision-language tasks.
    """

    # This class will not use the RotaryEmbedding to
    # apply rope to the query, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: Llama3RotaryEmbedding

    def __init__(
        self,
        *,
        rope: Llama3RotaryEmbedding,
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
            devices: Device to place the weights and run the computation. This is a distributed
                attention layer, so we use all devices during attention computation.
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
        self.hidden_size = hidden_size
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        self.devices = devices or [DeviceRef.CPU()]

        self._sharding_strategy: ShardingStrategy | None = None

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
        kv_collection: PagedCacheValues,
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

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        num_devices = sharding_strategy.num_devices

        if sharding_strategy.is_replicate:
            self.q_proj.sharding_strategy = sharding_strategy
            self.k_proj.sharding_strategy = sharding_strategy
            self.v_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )

        else:
            raise ValueError(
                "Qwen25VLDecoderAttentionWithRope only supports tensor parallel and replicate sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Qwen25VLDecoderAttentionWithRope]:
        """Creates sharded views of this attention layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3Attention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Qwen25VLDecoderAttentionWithRope layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        q_proj_shards = self.q_proj.shard(devices)
        k_proj_shards = self.k_proj.shard(devices)
        v_proj_shards = self.v_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            # Calculate sharded dimensions - handle uneven head distribution
            # Calculate the number of heads for this device
            head_start, head_end = _compute_shard_range(
                self.n_heads, shard_idx, len(self.devices)
            )
            sharded_num_heads = head_end - head_start

            sharded_head_start, sharded_head_end = _compute_shard_range(
                self.kv_params.n_kv_heads,
                shard_idx,
                len(self.devices),
            )
            sharded_num_kv_heads = sharded_head_end - sharded_head_start

            # Create new attention instance with sharded configuration
            sharded = Qwen25VLDecoderAttentionWithRope(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                dtype=self.q_proj.weight.dtype,
                devices=[device],
                linear_cls=self.o_proj.__class__,
                scale=self.scale,
                has_bias=self.has_bias,
            )

            # Assign sharded weights
            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            shards.append(sharded)

        return shards


class Qwen25VLDecoderTransformerBlock(Module):
    """Qwen2.5VL decoder transformer block customized for supporting 2D position ids."""

    def __init__(
        self,
        attention: Qwen25VLDecoderAttentionWithRope,
        mlp: ShardableCallable,
        attention_norm: ShardableCallable,
        mlp_norm: ShardableCallable,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.self_attn_shards = attention.shard(devices)

        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = attention_norm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = attention_norm.shard(devices)

        self.post_attention_layernorm = mlp_norm

        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = mlp_norm.shard(devices)

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        position_ids: TensorValue,
        mrope_section: list[int],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        attn_out = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
                # TODO: how to pass position_ids and mrope_section to each shard?
                position_ids=position_ids,
                mrope_section=mrope_section,
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_outs = self.allreduce(attn_out, signal_buffers)

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Apply post attention layer norm to each shard
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class Qwen25VLDecoder(Module):
    """Qwen2.5VL decoder model with support for vision-language tasks.

    This decoder implements the Qwen2.5VL architecture with:
    - Multi-axis rotary position embeddings (mrope) for 2D position encoding
    """

    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.devices = config.devices

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.kv_params.head_dim,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
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
            raise ValueError(f"Unsupported norm method: {config.norm_method}")

        # Select linear layer class.
        linear_cls: Callable[..., Linear]

        linear_cls = functools.partial(Linear)

        attention_cls: Callable[..., Qwen25VLDecoderAttentionWithRope]

        attention_cls = functools.partial(
            Qwen25VLDecoderAttentionWithRope,
            scale=config.attention_multiplier,
            has_bias=config.attention_bias,
        )

        layers = [
            Qwen25VLDecoderTransformerBlock(
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
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=config.model_quantization_encoding,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                devices=config.devices,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype

        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=None,
        )

        output = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=None,
            tied_weight=(
                embedding_layer.weight if config.tie_word_embeddings else None
            ),
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = LayerList(layers)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = output
        self.embed_tokens = embedding_layer
        self.kv_params = config.kv_params
        self.rope = rope
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValueLike,
        return_n_logits: TensorValue,
        image_embeddings: list[TensorValue],
        scatter_indices: list[TensorValue],
        gather_indices: list[TensorValue],
        position_ids: TensorValue,
        mrope_section: list[int],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        # Merge image embeddings into text embeddings.
        # Let the kernel handle the no-image embeddings case.
        # And use the first device's image embeddings since they're replicated.
        h = [
            merge_multimodal_embeddings_with_gather(
                inputs_embeds=h_device,
                multimodal_embeddings=img_embed,
                scatter_indices=scatter_indices,
                gather_indices=gather_indices,
            )
            for h_device, img_embed, scatter_indices, gather_indices in zip(
                h,
                image_embeddings,
                scatter_indices,
                gather_indices,
                strict=True,
            )
        ]

        # Create position embeddings shared across the decoder layers.
        freqs_cis = distribute_value(self.rope.freqs_cis, self.devices)

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=DeviceRef.CPU()
            )
            h = layer(
                layer_idx_tensor,
                h,
                kv_collections,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
                position_ids=position_ids,
                mrope_section=mrope_section,
                signal_buffers=signal_buffers,
            )

        # Retrieve a variable number of tokens
        last_token_indices = [offsets[1:] - 1 for offsets in input_row_offsets]
        assert h is not None and len(h) == len(last_token_indices)
        last_token_h = [
            ops.gather(h_device, indices, axis=0)
            for h_device, indices in zip(h, last_token_indices, strict=True)
        ]
        last_logits = ops.cast(
            # Take only the device 0 logits to device-to-host transfer.
            self.lm_head(
                [
                    self.norm_shards[i](last_token_h[i])
                    for i in range(len(last_token_h))
                ],
                signal_buffers,
            )[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_range = ops.range(
                return_n_logits[0],
                ops.constant(0, DType.int64, device=self.devices[0]),
                ops.constant(-1, DType.int64, device=self.devices[0]),
                out_dim="return_n_logits_range",
                device=self.devices[0],
                dtype=DType.int64,
            )

            last_indices = [
                ops.reshape(
                    ops.unsqueeze(row_offset[1:], -1) - return_range,
                    shape=(-1,),
                )
                for row_offset in input_row_offsets
            ]

            # Gather, normalize, and get logits
            variable_tokens = [
                self.norm_shards[i](ops.gather(h_device, indices, axis=0))
                for i, (h_device, indices) in enumerate(
                    zip(h, last_indices, strict=True)
                )
            ]
            logits = ops.cast(
                self.lm_head(variable_tokens, signal_buffers)[0], DType.float32
            )
            offsets = ops.range(
                0,
                last_indices[0].shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.devices[0],
            )
        elif self.return_logits == ReturnLogits.ALL:
            # Apply normalization to all hidden states and get all logits
            all_normalized = [
                self.norm_shards[i](h_device) for i, h_device in enumerate(h)
            ]
            logits = ops.cast(
                self.lm_head(all_normalized, signal_buffers)[0], DType.float32
            )
            offsets = input_row_offsets[0]

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
