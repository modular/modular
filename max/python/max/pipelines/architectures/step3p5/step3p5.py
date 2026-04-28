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
"""Step-3.5-Flash model implementation.

Features:
- Mixed attention: full + sliding window (per-layer)
- Per-layer RoPE theta and partial rotary factors
- MoE with shared expert, sigmoid router, router bias
- Zero-centered RMSNorm (weight_offset=1)
- Head-wise attention gate (g_proj with sigmoid)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.comm import Signals
from max.nn.comm.allreduce import Allreduce
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.moe import MoE
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)

from .layers.attention import Step3p5Attention
from .layers.moe_gate import Step3p5MoEGate
from .model_config import Step3p5Config


class _PartialRotaryEmbedding(Llama3RotaryEmbedding):
    """RoPE that only rotates the first ``rotary_dim`` dimensions.

    Produces a full ``head_dim``-sized ``freqs_cis`` tensor where the
    first ``rotary_dim`` dimensions carry real rotation frequencies and
    the remaining dimensions are identity (cos=1, sin=0) via zero
    frequencies.  This avoids the kernel's interleaved-only partial-RoPE
    constraint.
    """

    rotary_dim: int

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int,
        rotary_dim: int,
        interleaved: bool,
        scaling_params: Llama3RopeScalingParams | None,
    ) -> None:
        # Construct at full head_dim so freqs_cis has the right width.
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=interleaved,
            scaling_params=scaling_params,
        )
        self.rotary_dim = rotary_dim

    def _compute_inv_freqs(self) -> TensorValue:
        # Compute frequencies for only rotary_dim/2 blocks.
        n = self.rotary_dim
        iota = ops.range(
            0, n, step=2, dtype=DType.float64, device=DeviceRef.CPU()
        )
        active = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        # Apply Llama3 scaling to active freqs BEFORE zero-padding,
        # since the scaling divides by inv_freqs (would be div-by-zero
        # on the padded zeros).
        if self.scaling_params is not None:
            active = self._apply_scaling(active)
        # Zero-pad to full head_dim // 2
        pad_count = (self.head_dim - self.rotary_dim) // 2
        if pad_count > 0:
            zeros = ops.constant(
                [0.0] * pad_count, DType.float32, device=DeviceRef.CPU()
            )
            active = ops.concat((active, zeros))
        return active

    def _apply_scaling(self, inv_freqs: TensorValue) -> TensorValue:
        # Override required: scaling must happen BEFORE zero-padding (the parent
        # applies it inside _compute_inv_freqs which we also override).  The
        # parent's _apply_scaling is not directly reusable because it operates
        # on the full inv_freqs tensor, but here we only have the active
        # (non-padded) portion.
        sp = self.scaling_params
        assert sp is not None
        low_freq_wavelen = sp.orig_max_position / sp.low_freq_factor
        high_freq_wavelen = sp.orig_max_position / sp.high_freq_factor
        wave_len = 2 * math.pi / inv_freqs
        if sp.low_freq_factor != sp.high_freq_factor:
            smooth = (sp.orig_max_position / wave_len - sp.low_freq_factor) / (
                sp.high_freq_factor - sp.low_freq_factor
            )
        else:
            smooth = ops.constant(0, DType.float32, device=DeviceRef.CPU())
        return ops.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            ops.where(
                wave_len > low_freq_wavelen,
                inv_freqs / sp.factor,
                (1 - smooth) * inv_freqs / sp.factor + smooth * inv_freqs,
            ),
        )


class Step3p5TransformerBlock(Module):
    """Transformer block for Step-3.5 with mixed attention and MoE/MLP."""

    def __init__(
        self,
        config: Step3p5Config,
        layer_idx: int,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.devices = config.devices
        num_devices = len(config.devices)

        # Determine if this is a sliding window layer
        is_sliding = False
        if config.layer_types and layer_idx < len(config.layer_types):
            is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        # Select num heads based on layer type
        if is_sliding:
            num_heads = config.sliding_num_attention_heads
            num_kv_heads = config.sliding_num_attention_groups
        else:
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_attention_groups

        # Create attention layer
        self.self_attn = Step3p5Attention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
            sliding_window=config.sliding_window,
            use_head_wise_attn_gate=config.use_head_wise_attn_gate,
            dtype=config.dtype,
            rope=rope,
            linear_cls=linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            norm_dtype=config.norm_dtype or config.dtype,
            qk_norm_eps=config.rms_norm_eps or 1e-5,
        )
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx, linear_cls)
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
        self.mlp_shards = self.mlp.shard(config.devices)

        # Layer norms (zero-centered: weight_offset=1.0)
        self.input_layernorm = create_norm()
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = create_norm()
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

        self.allreduce = Allreduce(num_accelerators=num_devices)

    def _get_mlp(
        self,
        config: Step3p5Config,
        layer_idx: int,
        linear_cls: Callable[..., Linear],
    ) -> MLP | Step3p5MoEWithSharedExpert:
        """Get MLP or MoE layer based on config and layer index."""
        if layer_idx in config.moe_layers:
            swiglu_limit = (
                config.swiglu_limits[layer_idx]
                if layer_idx < len(config.swiglu_limits)
                else 0.0
            )
            swiglu_limit_shared = (
                config.swiglu_limits_shared[layer_idx]
                if layer_idx < len(config.swiglu_limits_shared)
                else 0.0
            )
            return Step3p5MoEWithSharedExpert(
                devices=config.devices,
                config=config,
                linear_cls=linear_cls,
                swiglu_limit=swiglu_limit,
                swiglu_limit_shared=swiglu_limit_shared,
            )
        else:
            return MLP(
                config.dtype,
                config.model_quantization_encoding,
                config.hidden_size,
                config.intermediate_size,
                config.devices,
                linear_cls,
            )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        # Input layer norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Self-attention
        attn_outs = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]

        # Allreduce attention outputs
        if len(self.devices) > 1:
            attn_outs = self.allreduce(attn_outs, signal_buffers)

        # Residual connection
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # MLP/MoE
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        # Allreduce MLP outputs
        if len(self.devices) > 1:
            mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual connection
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class Step3p5MoEWithSharedExpert(Module):
    """MoE layer with a shared (always-on) expert, using sigmoid routing.

    output = routed_moe(x) + shared_expert(x)

    Supports optional SwiGLU activation clipping (swiglu_limits from the
    paper) to prevent numerical blow-up at long contexts.
    """

    def __init__(
        self,
        *,
        devices: list[DeviceRef],
        swiglu_limit: float = 0.0,
        swiglu_limit_shared: float = 0.0,
        config: Step3p5Config | None = None,
        linear_cls: Callable[..., Linear] | None = None,
        is_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.devices = devices
        self.swiglu_limit = swiglu_limit
        self.swiglu_limit_shared = swiglu_limit_shared

        if is_sharding:
            return

        if config is None or linear_cls is None:
            raise ValueError(
                "config and linear_cls are required when is_sharding=False"
            )

        # Routed MoE
        self.moe = MoE(
            devices=self.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.moe_num_experts,
            num_experts_per_token=config.moe_top_k,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                Step3p5MoEGate,
                routed_scaling_factor=config.moe_router_scaling_factor,
                norm_topk_prob=config.norm_expert_weight,
            ),
            dtype=config.dtype,
            swiglu_limit=swiglu_limit,
        )

        # Shared expert (always-on MLP)
        self.share_expert = MLP(
            config.dtype,
            config.model_quantization_encoding,
            config.hidden_size,
            config.share_expert_dim,
            self.devices,
            linear_cls,
            swiglu_limit=swiglu_limit_shared,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        routed = self.moe(x)
        shared = self.share_expert(x)
        return routed + shared

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.moe.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.moe.sharding_strategy = strategy
        self.share_expert.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Step3p5MoEWithSharedExpert]:
        devices_list = list(devices)
        moe_shards = self.moe.shard(devices_list)
        share_expert_shards = self.share_expert.shard(devices_list)

        shards: list[Step3p5MoEWithSharedExpert] = []
        for i, device in enumerate(devices_list):
            # Bypass __init__ to avoid creating sub-modules that would
            # be immediately replaced by the already-sharded versions.
            sharded = Step3p5MoEWithSharedExpert(
                devices=[device],
                swiglu_limit=self.swiglu_limit,
                swiglu_limit_shared=self.swiglu_limit_shared,
                is_sharding=True,
            )
            sharded.moe = moe_shards[i]
            sharded.share_expert = share_expert_shards[i]
            shards.append(sharded)
        return shards


class Step3p5(DistributedLogitsPostprocessMixin, Module):
    """Step-3.5-Flash model."""

    def __init__(self, config: Step3p5Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Step-3.5 is not implemented yet")
        if config.model_quantization_encoding is not None:
            raise NotImplementedError("GGUFQ Step-3.5 is not implemented yet")

        # Per-layer RoPE configuration.
        # Step-3.5 uses per-layer rope_theta and partial rotary factors:
        #   - Full attention: partial_rotary_factor=0.5 (64/128 dims), with
        #     rope_scaling
        #   - SWA: partial_rotary_factor=1.0 (128/128 dims), no rope_scaling
        # We store RoPE objects keyed by (theta, rotary_dim, use_scaling) and
        # a per-layer mapping.  freqs_cis access is deferred to __call__
        # (inside the Graph context) since RoPE computation emits graph ops.
        full_head_dim = config.kv_params.head_dim
        num_layers = config.num_hidden_layers
        yarn_only = set(config.yarn_only_types)

        per_layer_thetas = (
            config.per_layer_rope_theta or [config.rope_theta] * num_layers
        )
        per_layer_prf = config.partial_rotary_factors or [1.0] * num_layers

        rope_cache: dict[tuple[float, int, bool], Llama3RotaryEmbedding] = {}
        layer_rope_keys: list[tuple[float, int, bool]] = []
        for i in range(num_layers):
            theta = (
                per_layer_thetas[i]
                if i < len(per_layer_thetas)
                else config.rope_theta
            )
            prf = per_layer_prf[i] if i < len(per_layer_prf) else 1.0
            rotary_dim = int(full_head_dim * prf)
            layer_type = (
                config.layer_types[i]
                if config.layer_types and i < len(config.layer_types)
                else "full_attention"
            )
            use_scaling = layer_type in yarn_only
            key = (theta, rotary_dim, use_scaling)
            if key not in rope_cache:
                if rotary_dim < full_head_dim:
                    rope_cache[key] = _PartialRotaryEmbedding(
                        dim=config.hidden_size,
                        n_heads=config.num_attention_heads,
                        theta=theta,
                        max_seq_len=config.max_seq_len,
                        head_dim=full_head_dim,
                        rotary_dim=rotary_dim,
                        interleaved=config.interleaved_rope_weights,
                        scaling_params=(
                            config.rope_scaling_params if use_scaling else None
                        ),
                    )
                else:
                    rope_cache[key] = Llama3RotaryEmbedding(
                        dim=config.hidden_size,
                        n_heads=config.num_attention_heads,
                        theta=theta,
                        max_seq_len=config.max_seq_len,
                        head_dim=full_head_dim,
                        interleaved=config.interleaved_rope_weights,
                        scaling_params=(
                            config.rope_scaling_params if use_scaling else None
                        ),
                    )
            layer_rope_keys.append(key)

        self._rope_cache = rope_cache
        self._layer_rope_keys = layer_rope_keys
        # Keep a rope reference for the interleaved flag used by attention.
        first_key = layer_rope_keys[0]
        self.rope = rope_cache[first_key]

        # Zero-centered RMSNorm factory (weight_offset=1.0)
        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "Step-3.5 requires RMSNorm. Set norm_method='rms_norm' and "
                "provide rms_norm_eps."
            )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        # Transformer layers (all share self.rope for the interleaved flag;
        # actual per-layer freqs_cis is passed at call time via _layer_freqs).
        self.layers = LayerList(
            [
                Step3p5TransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=self.rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final norm (zero-centered)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and output layers
        embedding_dtype = config.dtype
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_dtype = config.quant_config.embedding_output_dtype

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_dtype,
            config.devices,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_dtype,
            devices=config.devices,
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
    ) -> tuple[TensorValue, ...]:
        # Embeddings
        h = self.embed_tokens(tokens, signal_buffers)

        # Build per-layer freqs_cis inside the graph context (deferred
        # from __init__ because RoPE emits graph ops).
        _freqs_by_key: dict[tuple[float, int, bool], list[TensorValue]] = {}
        per_layer_freqs: list[list[TensorValue]] = []
        for key in self._layer_rope_keys:
            if key not in _freqs_by_key:
                fc = self._rope_cache[key].freqs_cis
                _freqs_by_key[key] = [fc.to(device) for device in self.devices]
            per_layer_freqs.append(_freqs_by_key[key])

        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # Transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx = ops.constant(idx, DType.uint32, device=DeviceRef.CPU())
            h = layer(
                layer_idx,
                h,
                kv_collections,
                per_layer_freqs[idx],
                input_row_offsets_list,
                signal_buffers,
            )

        return self._postprocess_logits(
            h, input_row_offsets_list, return_n_logits, signal_buffers
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]

        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()

        flattened_kv_types = kv_inputs.flatten()
        return tuple(base_inputs + signal_buffer_types + flattened_kv_types)
