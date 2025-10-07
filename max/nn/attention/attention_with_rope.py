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

"""An opaque KV Cache optimized attention mechanism with Rope."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weight import _compute_shard_range
from max.nn.float8_config import Float8Config
from max.nn.kernels import convert_weights_to_fp8_fnuz_if_needed

from ..clamp import clamp
from ..comm import Allreduce
from ..kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    fused_qkv_ragged_matmul_quantized,
    fused_qkv_ragged_matmul_scaled_float8,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
    unfused_qkv_ragged_matmul_gguf_quantized,
)
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Module, Shardable
from ..linear import Linear
from ..no_opaque_kernels import (
    PagedKVCacheTensorsNoOpaque,
    flash_attention_ragged_no_opaque,
    rope_no_opaque,
    store_k_cache,
    store_v_cache,
)
from ..rotary_embedding import RotaryEmbedding
from .interfaces import (
    AttentionImpl,
    AttentionImplQKV,
    DistributedAttentionImpl,
)
from .mask_config import MHAMaskVariant


@dataclass
class AttentionWithRopeV1(AttentionImpl):
    """Implementation of attention that uses the rope frequency.

    Deprecated: Use `AttentionWithRope` instead.
    """

    # This class does not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
    rope: RotaryEmbedding
    bias: TensorValue | None = None
    perm_idx: TensorValue | None = None
    quantization_config: QuantizationConfig | None = None

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        if self.quantization_config:
            xq = fused_qkv_ragged_matmul_quantized(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                bias=self.bias,
                perm_idx=self.perm_idx,
                quantization_config=self.quantization_config,
            )
        else:
            xq = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                bias=self.bias,
            )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
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

        return self.wo(attn_out)


class AttentionWithRope(Module, Shardable):
    """Implementation of attention that uses Rotary Position Embedding (RoPE)."""

    # This class will not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        sharding_strategy: ShardingStrategy | None = None,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        float8_config: Float8Config | None = None,
        clip_qkv: float | None = None,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            sharding_strategy: Optional initial sharding strategy.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            dtype: DType of the QKV and output projection weights.
            devices: Device(s) on which to place the weights and run the computation. If multiple are
                provided, the first device is used for weight placement here.
            linear_cls: Linear class to use for projections.
            stacked_qkv: Whether Q/K/V weights are stacked in a single Weight.
            scale: Optional attention scale; defaults to sqrt(1/head_dim).
            has_bias: Whether Q/K/V have bias (stacked_qkv forbids bias).
            float8_config: Optional Float8 config (dynamic or static).
            clip_qkv: If provided, clamp Q/K/V weights to [-clip_qkv, clip_qkv].
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_params = kv_params
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.clip_qkv = clip_qkv
        self.devices = devices or [DeviceRef.CPU()]
        self.float8_config = float8_config
        self.stacked_qkv = stacked_qkv
        self._sharding_strategy: ShardingStrategy | None = None

        if stacked_qkv and clip_qkv:
            raise ValueError(
                "`clip_qkv` not yet supported when `stacked_qkv=True`."
            )

        if stacked_qkv and has_bias:
            raise ValueError("Bias is not supported with stacked_qkv.")

        # Static FP8 + stacked QKV needs special scale plumbing; not wired up yet.
        if (
            stacked_qkv
            and (float8_config is not None)
            and float8_config.is_static
        ):
            raise NotImplementedError(
                "Float8 static scaling with stacked_qkv=True is not supported yet."
            )

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy is not supported in the Attention layer."
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_weight_dim

        if stacked_qkv:
            # Keep names consistent with other stacks by suffixing ".weight".
            self.qkv_proj = Weight(
                name="qkv_proj.weight",
                dtype=dtype,
                shape=[q_weight_dim + 2 * kv_weight_dim, hidden_size],
                device=self.devices[0],
            )
        else:
            self.q_proj = linear_cls(
                in_dim=hidden_size,
                out_dim=q_weight_dim,
                dtype=dtype,
                device=self.devices[0],
                has_bias=has_bias,
                float8_config=float8_config,
            )
            self.k_proj = linear_cls(
                in_dim=hidden_size,
                out_dim=kv_weight_dim,
                dtype=dtype,
                device=self.devices[0],
                has_bias=has_bias,
                float8_config=float8_config,
            )
            self.v_proj = linear_cls(
                in_dim=hidden_size,
                out_dim=kv_weight_dim,
                dtype=dtype,
                device=self.devices[0],
                has_bias=has_bias,
                float8_config=float8_config,
            )

        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            float8_config=float8_config,
        )

        if sharding_strategy is not None:
            self.sharding_strategy = sharding_strategy

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Module sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the Module sharding strategy and propagate to weights.

        We support both tensor-parallel and data-parallel (replicate) sharding strategies.
        """
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy

            num_devices = strategy.num_devices
            if self.stacked_qkv:
                # Partition the [Q|K|V] block by heads.
                self.qkv_proj.sharding_strategy = ShardingStrategy.stacked_qkv(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            else:
                # Column-shard by output channels (heads) for each projection.
                self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
                self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
                self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )

            # Row-shard o_proj.weight (standard tensor parallel o-proj).
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
        elif strategy.is_replicate:
            self._sharding_strategy = strategy

            num_devices = strategy.num_devices
            if self.stacked_qkv:
                self.qkv_proj.sharding_strategy = ShardingStrategy.replicate(
                    num_devices
                )
            else:
                self.q_proj.sharding_strategy = ShardingStrategy.replicate(
                    num_devices
                )
                self.k_proj.sharding_strategy = ShardingStrategy.replicate(
                    num_devices
                )
                self.v_proj.sharding_strategy = ShardingStrategy.replicate(
                    num_devices
                )
            self.o_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError(
                "Only tensor-parallel (rowwise) or data-parallel (replicate) sharding strategies are supported for AttentionWithRope"
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[AttentionWithRope]:
        """Create sharded views across `devices` (tensor-parallel).

        Returns one `AttentionWithRope` per device with appropriately sliced weights.
        """
        devices = list(devices)
        if not self.sharding_strategy:
            raise ValueError(
                "AttentionWithRope cannot be sharded: no sharding_strategy set. "
                "Set `self.sharding_strategy = ShardingStrategy.tensor_parallel(N)` first."
            )

        if self.sharding_strategy.is_tensor_parallel:
            if DeviceRef.CPU() in devices:
                raise ValueError(
                    "Tensor-parallel AttentionWithRope does not support CPU devices"
                )

            if self.stacked_qkv:
                qkv_proj_shards = self.qkv_proj.shard(devices)
            else:
                q_proj_shards = self.q_proj.shard(devices)
                k_proj_shards = self.k_proj.shard(devices)
                v_proj_shards = self.v_proj.shard(devices)
            o_proj_shards = self.o_proj.shard(devices)

            default_dtype = o_proj_shards[0].weight.dtype
            linear_cls = self.o_proj.__class__

            shards: list[AttentionWithRope] = []
            num_devices = len(devices)
            for n, device in enumerate(devices):
                # Compute this shard's number of attention heads.
                head_start, head_end = _compute_shard_range(
                    self.n_heads, n, num_devices
                )
                device_num_heads = head_end - head_start

                layer = AttentionWithRope(
                    rope=self.rope,
                    num_attention_heads=device_num_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    devices=[device],
                    dtype=default_dtype,
                    linear_cls=linear_cls,
                    stacked_qkv=self.stacked_qkv,
                    scale=self.scale,
                    has_bias=self.has_bias,
                    float8_config=self.float8_config,
                    clip_qkv=self.clip_qkv,
                )

                if self.stacked_qkv:
                    layer.qkv_proj = qkv_proj_shards[n]
                else:
                    layer.q_proj = q_proj_shards[n]
                    layer.k_proj = k_proj_shards[n]
                    layer.v_proj = v_proj_shards[n]
                layer.o_proj = o_proj_shards[n]

                shards.append(layer)
            return shards

        elif self.sharding_strategy.is_replicate:
            # Replicate full weights to each device (no head split).
            if self.stacked_qkv:
                qkv_proj_replicas = self.qkv_proj.shard(devices)
            else:
                q_proj_replicas = self.q_proj.shard(devices)
                k_proj_replicas = self.k_proj.shard(devices)
                v_proj_replicas = self.v_proj.shard(devices)
            o_proj_replicas = self.o_proj.shard(devices)

            default_dtype = o_proj_replicas[0].weight.dtype
            linear_cls = self.o_proj.__class__

            replicas: list[AttentionWithRope] = []
            for i, device in enumerate(devices):
                replica = AttentionWithRope(
                    rope=self.rope,
                    num_attention_heads=self.n_heads,  # DP keeps full heads
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    devices=[device],
                    dtype=default_dtype,
                    linear_cls=linear_cls,
                    stacked_qkv=self.stacked_qkv,
                    scale=self.scale,
                    has_bias=self.has_bias,
                    float8_config=self.float8_config,
                    clip_qkv=self.clip_qkv,
                )
                if self.stacked_qkv:
                    replica.qkv_proj = qkv_proj_replicas[i]
                else:
                    replica.q_proj = q_proj_replicas[i]
                    replica.k_proj = k_proj_replicas[i]
                    replica.v_proj = v_proj_replicas[i]
                replica.o_proj = o_proj_replicas[i]
                replicas.append(replica)
            return replicas

        else:
            # Should not happen due to setter validation.
            raise ValueError(
                "Unsupported sharding strategy for AttentionWithRope"
            )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj
        else:
            wq: TensorValue = self.q_proj.weight
            wk: TensorValue = self.k_proj.weight
            wv: TensorValue = self.v_proj.weight
            if self.clip_qkv:
                wq = clamp(wq, min=-self.clip_qkv, max=self.clip_qkv)
                wk = clamp(wk, min=-self.clip_qkv, max=self.clip_qkv)
                wv = clamp(wv, min=-self.clip_qkv, max=self.clip_qkv)

            # Here we are rescaling the weights to be based on the max scale.
            # This feels super fishy and like it could greatly hurt accuracy.
            # That said, for these float8 models, all models run with vllm
            # (not supported by torch/transformers). As such, vllm is the
            # canonical implementation for correctness. This rescaling is what
            # vllm does.
            # https://github.com/vllm-project/vllm/blob/9b1769dd9ad13a5688d1e2b1b5f00b07b3716969/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L35
            if (
                self.float8_config
                and self.float8_config.weight_scale.is_tensor
                and self.q_proj.weight_scale is not None
                and self.k_proj.weight_scale is not None
                and self.v_proj.weight_scale is not None
            ):
                wq = wq * self.q_proj.weight_scale.to(wq.device)
                wk = wk * self.k_proj.weight_scale.to(wk.device)
                wv = wv * self.v_proj.weight_scale.to(wv.device)

            wqkv = ops.concat((wq, wk, wv))
            if self.float8_config and self.float8_config.is_static:
                assert self.qkv_weight_scale is not None

                wqkv, qkv_weight_scale = convert_weights_to_fp8_fnuz_if_needed(
                    wqkv, self.qkv_weight_scale.to(DeviceRef.CPU())
                )
                wqkv = quantize_static_scaled_float8(
                    wqkv, qkv_weight_scale.to(DeviceRef.CPU())
                )

            return wqkv

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        # This was already checked in the constructor.
        assert not self.stacked_qkv

        # Access bias, which should all exist since has_bias=True.
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        )

    @property
    def qkv_input_scale(self) -> TensorValue | None:
        """The max of q, k, and v scale input vectors."""
        if not self.float8_config or self.float8_config.is_dynamic:
            return None

        # TODO(MODELS-595): Reuse AttentionWithRopeQKV for this.
        if self.stacked_qkv:
            raise NotImplementedError(
                "QKV input scale not implemented for stacked_qkv=True"
            )

        assert self.q_proj.input_scale is not None
        assert self.k_proj.input_scale is not None
        assert self.v_proj.input_scale is not None

        return ops.max(
            ops.concat(
                (
                    self.q_proj.input_scale.reshape((1,)),
                    self.k_proj.input_scale.reshape((1,)),
                    self.v_proj.input_scale.reshape((1,)),
                )
            )
        ).reshape(())

    @property
    def qkv_weight_scale(self) -> TensorValue:
        """The max of q, k, and v scale weight vectors."""
        assert self.float8_config is not None

        # TODO(MODELS-595): Reuse AttentionWithRopeQKV for this.
        if self.stacked_qkv:
            # TODO: Handle stacked QKV weight scale when implemented
            raise NotImplementedError(
                "QKV weight scale not implemented for stacked_qkv=True"
            )

        assert self.q_proj.weight_scale is not None
        assert self.k_proj.weight_scale is not None
        assert self.v_proj.weight_scale is not None

        q_scale: TensorValue = self.q_proj.weight_scale
        k_scale: TensorValue = self.k_proj.weight_scale
        v_scale: TensorValue = self.v_proj.weight_scale
        if len(q_scale.shape) == 0:
            q_scale = q_scale.reshape((1,))
        if len(k_scale.shape) == 0:
            k_scale = k_scale.reshape((1,))
        if len(v_scale.shape) == 0:
            v_scale = v_scale.reshape((1,))

        weight_scale = ops.concat((q_scale, k_scale, v_scale))

        if self.float8_config.is_dynamic:
            # In the dynamic scaling case, return the weight scales directly.
            return weight_scale

        # Static case: return a scalar max QKV weight scale.
        return ops.max(weight_scale).reshape([])

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = self.wqkv.to(x.device)
        wqkv_bias = (
            self.wqkv_bias.to(x.device) if self.wqkv_bias is not None else None
        )

        if self.float8_config:
            # FP8 path
            weight_scale = self.qkv_weight_scale
            if self.float8_config.is_static:
                assert self.qkv_input_scale is not None
                x = quantize_static_scaled_float8(
                    x, self.qkv_input_scale.to(DeviceRef.CPU())
                )
                x_scales = self.qkv_input_scale
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    self.float8_config.input_scale,
                    self.float8_config.weight_scale,
                    scales_type=weight_scale.dtype,
                )

            xq = fused_qkv_ragged_matmul_scaled_float8(
                self.kv_params,
                input=x,
                wqkv=wqkv,
                bias=wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale.to(x.device),
            )
        else:
            # Regular fused QKV matmul.
            xq = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x,
                wqkv=wqkv,
                bias=wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
            )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = ops.reshape(
            attn_out, shape=[total_seq_len, self.q_weight_dim]
        )
        return self.o_proj(attn_out)


class GGUFQAttentionWithRope(AttentionWithRope):
    """Implementation of attention with GGUF quantized weights."""

    # This class will not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ) -> None:
        """Initializes the GGUF attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the weights, should always be uint8.
            devices: Device(s) on which to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            quantization_encoding: Quantization encoding of the weights.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """
        # Skip AttentionWithRope.__init__ because the weights are created differently.
        Module.__init__(self)

        if dtype != DType.uint8:
            raise ValueError(
                f"GGUFQAttentionWithRope only supports uint8 dtype weights but got {dtype}"
            )
        if clip_qkv is not None:
            raise ValueError(
                "clip_qkv is not supported for GGUFQAttentionWithRope"
            )
        if has_bias:
            raise ValueError("GGUFQAttentionWithRope does not support bias")
        if not quantization_encoding.is_gguf:
            raise ValueError(
                f"Only GGUF quantization encoding is supported for GGUFQAttentionWithRope. Found: {quantization_encoding}"
            )
        if not kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{kv_params.cache_strategy} cache strategy is not supported in the Attention layer."
            )

        self.quantization_encoding = quantization_encoding
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.devices = devices or [DeviceRef.CPU()]

        self.q_proj_weight = Weight(
            name="q_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.k_proj_weight = Weight(
            name="k_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.v_proj_weight = Weight(
            name="v_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=1,  # Shape will be overridden at load_state_dict.
            out_dim=1,  # Shape will be overridden at load_state_dict.
            dtype=DType.uint8,
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        raise NotImplementedError(
            "wqkv is not implemented for unfused GGUFQAttentionWithRope"
        )

    @property
    def wqkv_bias(self) -> TensorValue | None:
        raise NotImplementedError(
            "wqkv_bias is not implemented for unfused GGUFQAttentionWithRope"
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        assert self.q_proj_weight.quantization_encoding is not None
        assert self.k_proj_weight.quantization_encoding is not None
        assert self.v_proj_weight.quantization_encoding is not None

        # Unfused GGUF path.
        xq = unfused_qkv_ragged_matmul_gguf_quantized(
            self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            n_heads=self.n_heads,
            q_weight=self.q_proj_weight,
            k_weight=self.k_proj_weight,
            v_weight=self.v_proj_weight,
            quantization_encoding_q=self.q_proj_weight.quantization_encoding,
            quantization_encoding_k=self.k_proj_weight.quantization_encoding,
            quantization_encoding_v=self.v_proj_weight.quantization_encoding,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
        )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        freqs_cis = ops.cast(freqs_cis, xq.dtype)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

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


class GPTQAttentionWithRope(AttentionWithRope):
    """Implementation of the GPTQ attention layer."""

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        scale: float | None = None,
        linear_cls: Callable[..., Linear] = Linear,
    ) -> None:
        # Skip AttentionWithRope.__init__ because the weights are created differently.
        Module.__init__(self)
        self.quantization_config = quantization_config
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.hidden_size = hidden_size
        self.devices = devices or [DeviceRef.CPU()]
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy is not supported in the Attention layer."
            )

        self.kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        self.q_proj_qweight = Weight(
            name="q_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.k_proj_qweight = Weight(
            name="k_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.v_proj_qweight = Weight(
            name="v_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )

        self.q_proj_scales = Weight(
            name="q_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.k_proj_scales = Weight(
            name="k_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.v_proj_scales = Weight(
            name="v_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )

        self.perm_idx = None
        if quantization_config.desc_act:
            self.perm_idx = Weight(
                name="q_proj.perm_idx",
                dtype=DType.int32,
                shape=[hidden_size],
                device=self.devices[0],
            )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors (packed + scales)."""

        # fmt: off
        # The `qweight` tensor for a QuantLinear is of type uint32. When allocated as bytes, we reshape the
        # uint8 tensor to [cols, rows * 4] so concatenating the uint8 tensors along axis=1 is equivalent to
        # concatenating the original uint32 tensors along axis=1.
        wq_qweight = ops.reshape(self.q_proj_qweight, (-1, self.hidden_size * 4))
        wk_qweight = ops.reshape(self.k_proj_qweight, (-1, self.kv_weight_dim * 4))
        wv_qweight = ops.reshape(self.v_proj_qweight, (-1, self.kv_weight_dim * 4))

        wqkv_qweight = ops.reshape(
            ops.concat((wq_qweight, wk_qweight, wv_qweight), axis=1),
            (-1, self.hidden_size + 2 * self.kv_weight_dim),
        )
        # `scales` tensor is in f16/bf16 type, so we reshape the uint8 tensor to [cols, rows * 2].
        wq_scales = ops.reshape(self.q_proj_scales, (-1, self.hidden_size * 2))
        wk_scales = ops.reshape(self.k_proj_scales, (-1, self.kv_weight_dim * 2))
        wv_scales = ops.reshape(self.v_proj_scales, (-1, self.kv_weight_dim * 2))

        wqkv_scales = ops.reshape(
            ops.concat((wq_scales, wk_scales, wv_scales), axis=1),
            (-1, self.hidden_size + 2 * self.kv_weight_dim),
        )
        # fmt: on
        return ops.concat((wqkv_qweight, wqkv_scales), axis=0)

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = self.wqkv
        if self.devices:
            wqkv = wqkv.to(self.devices[0])

        xq = fused_qkv_ragged_matmul_quantized(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            perm_idx=self.perm_idx,
            quantization_config=self.quantization_config,
        )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

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


def distribute_value(
    v: TensorValue, devices: list[DeviceRef]
) -> list[TensorValue]:
    return [v.to(device) for device in devices]


class TensorParallelAttentionWithRope(
    AttentionWithRope, DistributedAttentionImpl
):
    """Tensor-parallel wrapper that delegates sharding to the base module."""

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
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        float8_config: Float8Config | None = None,
        clip_qkv: float | None = None,
    ) -> None:
        """Initializes the distributed (tensor parallel) attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            devices: Device(s) on which to place the weights and run the computation. Must
                provide at least 2 devices for tensor parallel attention.
            dtype: DType of the QKV and output projection weights.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            float8_config: Float8 configuration for quantization.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`.
        """
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
            linear_cls=linear_cls,
            stacked_qkv=stacked_qkv,
            scale=scale,
            has_bias=has_bias,
            float8_config=float8_config,
            clip_qkv=clip_qkv,
        )
        if DeviceRef.CPU() in self.devices:
            raise ValueError(
                "TensorParallelAttentionWithRope does not support CPU devices"
            )

        num_devices = len(self.devices)
        self.allreduce = Allreduce(num_devices)

        # Delegate: configure base sharding + create per-device modules.
        self.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        x: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")
        if len(input_row_offsets) != len(self.devices):
            raise ValueError(
                f"Expected {len(self.devices)} input_row_offsets, got {len(input_row_offsets)}"
            )
        if not all(isinstance(t, TensorValue) for t in input_row_offsets):
            raise TypeError(
                "All elements in input_row_offsets must be TensorValue instances"
            )
        if not all(isinstance(t, TensorValue) for t in freqs_cis):
            raise TypeError(
                "All elements in freqs_cis must be TensorValue instances"
            )

        attn_outputs = [
            self.list_of_attentions[i](
                layer_idx,
                x[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i in range(len(self.devices))
        ]

        return self.allreduce(
            inputs=attn_outputs, signal_buffers=signal_buffers
        )


class DataParallelAttentionWithRope(AttentionWithRope):
    """Data-parallel implementation of Attention with RoPE.

    This replicates the attention module across devices and runs each replica on
    its local inputs (x, kv, freqs_cis, input_row_offsets). No collective ops
    are required; KV-cache remains local to each device.

    Notes:
      - Assumes the caller has already distributed `xs`, `kv_collections`,
        `freqs_cis`, and `input_row_offsets` so that index i corresponds to
        device i, with `input_row_offsets[i]` rebased to start at 0.
    """

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
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        float8_config: Float8Config | None = None,
        clip_qkv: float | None = None,
    ) -> None:
        super().__init__(
            rope=rope,
            sharding_strategy=None,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
            linear_cls=linear_cls,
            stacked_qkv=stacked_qkv,
            scale=scale,
            has_bias=has_bias,
            float8_config=float8_config,
            clip_qkv=clip_qkv,
        )
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)

        # Replicate component weights/modules to each device.
        if self.stacked_qkv:
            self.qkv_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            qkv_proj_replicas = self.qkv_proj.shard(self.devices)
        else:
            self.q_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            q_proj_replicas = self.q_proj.shard(self.devices)
            k_proj_replicas = self.k_proj.shard(self.devices)
            v_proj_replicas = self.v_proj.shard(self.devices)

        self.o_proj.sharding_strategy = ShardingStrategy.replicate(num_devices)
        o_proj_replicas = self.o_proj.shard(self.devices)

        # Build one full copy per device (no head-splitting).
        self.replicated_attentions: list[AttentionWithRope] = []
        for i, device in enumerate(self.devices):
            replica = AttentionWithRope(
                rope=self.rope,
                sharding_strategy=None,
                num_attention_heads=self.n_heads,  # DP keeps full heads
                num_key_value_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                devices=[device],
                dtype=dtype,
                linear_cls=linear_cls,
                stacked_qkv=self.stacked_qkv,
                scale=self.scale,
                has_bias=self.has_bias,
                float8_config=self.float8_config,
                clip_qkv=self.clip_qkv,
            )
            if self.stacked_qkv:
                replica.qkv_proj = qkv_proj_replicas[i]
            else:
                replica.q_proj = q_proj_replicas[i]
                replica.k_proj = k_proj_replicas[i]
                replica.v_proj = v_proj_replicas[i]
            replica.o_proj = o_proj_replicas[i]
            self.replicated_attentions.append(replica)

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, freqs_cis, and input_row_offsets must all have "
                f"length equal to number of devices ({n})"
            )

        outs: list[TensorValue] = []
        for i in range(n):
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                continue

            outs.append(
                self.replicated_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                )
            )
        return outs


@dataclass
class AttentionWithRopeQKV(AttentionImplQKV):
    # This class will not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
    rope: RotaryEmbedding

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]

        wqkv = ops.concat((self.wq, self.wk, self.wv))

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Cast freqs_cis to xq's dtype to match the fused_qk_ragged_rope kernel.
        freqs_cis = ops.cast(freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

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

        return self.wo(attn_out)


class AttentionWithRopeNoOpaque(Module):
    """Attention with RoPE without opaque KV cache.

    Assumes:
      - no float8
      - no stacked qkv
      - no bias
      - no clip_qkv
      - no float8_config
    """

    # This class will not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
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
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            dtype: DType of the QKV and output projection weights.
            devices: Device(s) on which to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.devices = devices or [DeviceRef.CPU()]

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        # To keep names consistent, suffix ".weight".
        self.qkv_proj = Linear(
            in_dim=hidden_size,
            out_dim=q_weight_dim + 2 * kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
        )

        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedKVCacheTensorsNoOpaque,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into QKV Matmul.
        # TODO:
        # - we might need to have individual Q, K, and V matmuls, that shouldn't matter
        x_qkv = self.qkv_proj(x)

        # Split QKV into Q, K, and V.
        q_end = self.n_heads * self.kv_params.head_dim
        k_end = q_end + self.kv_params.n_kv_heads * self.kv_params.head_dim
        v_end = k_end + self.kv_params.n_kv_heads * self.kv_params.head_dim
        x_q = x_qkv[:, :q_end]
        x_k = x_qkv[:, k_end:v_end]
        x_v = x_qkv[:, v_end:]

        freqs_cis = ops.cast(self.rope.freqs_cis, x_q.dtype)

        # call rope individual on Q and K. We could theoretically fuse these into a single kernel
        # TODO:
        # - this should just be inside of `RotaryEmbedding`
        # - should this be fused automatically? That class has an implementation.
        x_q = x_q.reshape((-1, self.n_heads, self.kv_params.head_dim))
        x_k = x_k.reshape(
            (-1, self.kv_params.n_kv_heads, self.kv_params.head_dim)
        )
        xq_rope = rope_no_opaque(
            x_q, input_row_offsets, kv_collection.cache_lengths, freqs_cis
        )
        xk_rope = rope_no_opaque(
            x_k, input_row_offsets, kv_collection.cache_lengths, freqs_cis
        )

        store_k_cache(kv_collection, xk_rope, input_row_offsets, layer_idx)
        store_v_cache(
            kv_collection,
            x_v.reshape(
                (-1, self.kv_params.n_kv_heads, self.kv_params.head_dim)
            ),
            input_row_offsets,
            layer_idx,
        )

        attn_out = flash_attention_ragged_no_opaque(
            self.kv_params,
            input=xq_rope,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)
