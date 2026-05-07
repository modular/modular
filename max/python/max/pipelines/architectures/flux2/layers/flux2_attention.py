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

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu, rope_ragged_with_position_ids
from max.nn.layer import LayerList, Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig

from ..model_config import Flux2BlockQuant
from .embeddings import get_1d_rotary_pos_embed


def _apply_flux2_qk_rope(
    query: TensorValue,
    key: TensorValue,
    cos: TensorValue,
    sin: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    num_heads = query.shape[2]
    head_dim = query.shape[3]

    query_ragged = ops.reshape(
        query, [batch_size * seq_len, num_heads, head_dim]
    )
    key_ragged = ops.reshape(key, [batch_size * seq_len, num_heads, head_dim])

    # Convert repeat-interleaved ([cos, cos], [sin, sin]) to [cos, sin] pairs.
    cos_pairs = ops.reshape(cos, [cos.shape[0], cos.shape[1] // 2, 2])[..., 0]
    sin_pairs = ops.reshape(sin, [sin.shape[0], sin.shape[1] // 2, 2])[..., 0]
    freqs_cis = ops.reshape(
        ops.stack([cos_pairs, sin_pairs], axis=-1),
        [cos.shape[0], cos.shape[1]],
    )
    position_ids = ops.range(
        0,
        seq_len,
        1,
        dtype=DType.uint32,
        device=query.device,
    )
    # broadcast_to instead of tile: tile has no GPU kernel and forces a
    # CPU round-trip. broadcast_to expands [1, seq_len] -> [batch_size, seq_len]
    # entirely on GPU.
    position_ids = ops.broadcast_to(
        ops.unsqueeze(position_ids, 0), [batch_size, seq_len]
    )
    position_ids = ops.reshape(position_ids, [batch_size * seq_len])

    query_out = rope_ragged_with_position_ids(
        query_ragged,
        freqs_cis,
        position_ids,
        interleaved=True,
    )
    key_out = rope_ragged_with_position_ids(
        key_ragged,
        freqs_cis,
        position_ids,
        interleaved=True,
    )
    return (
        ops.reshape(query_out, [batch_size, seq_len, num_heads, head_dim]),
        ops.reshape(key_out, [batch_size, seq_len, num_heads, head_dim]),
    )


class Flux2SwiGLU(Module):
    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Output tensor of shape [..., dim//2].
        """
        x1, x2 = ops.chunk(x, chunks=2, axis=-1)
        return ops.silu(x1) * x2


class Flux2FeedForward(Module, Shardable):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
        *,
        dtype: DType,
        devices: Sequence[DeviceRef],
        quant_config: QuantConfig | None = None,
    ):
        """Initialize Flux2FeedForward.

        Args:
            dim: Input dimension.
            dim_out: Output dimension (defaults to dim).
            mult: Multiplier for hidden dimension (defaults to 3.0).
            inner_dim: Explicit inner dimension (overrides mult if provided).
            bias: Whether to use bias in linear layers.
            dtype: Weight dtype.
            devices: Devices for placement and tensor parallelism. The
                un-sharded base lives on ``devices[0]``; ``shard()`` produces
                one shard per device.
            quant_config: Quantization config applied to both linears.
        """
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.has_bias = bias
        self.dtype = dtype
        self.devices = list(devices)
        self.quant_config = quant_config
        self.linear_in = Linear(
            dim,
            inner_dim * 2,
            dtype,
            self.devices[0],
            has_bias=bias,
            quant_config=quant_config,
        )
        self.act_fn = Flux2SwiGLU()
        self.linear_out = Linear(
            inner_dim,
            dim_out,
            dtype,
            self.devices[0],
            has_bias=bias,
            quant_config=quant_config,
        )
        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply feedforward transformation.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Output tensor of shape [..., dim_out].
        """
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if strategy.is_replicate:
            self.linear_in.sharding_strategy = strategy
            self.linear_out.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            # SwiGLU on a stacked-gate-up linear_in: each device must own
            # the matching halves of gate and up so silu(gate)*up runs
            # locally. ``gate_up`` axis=0 picks aligned slices from each
            # half rather than cutting the output dim contiguously.
            self.linear_in.sharding_strategy = ShardingStrategy.gate_up(
                strategy.num_devices, axis=0
            )
            # linear_out reduces along the inner_dim axis that was just
            # sharded; columnwise leaves partial sums for a block-level
            # allreduce.
            self.linear_out.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
        else:
            raise ValueError(
                "Flux2FeedForward only supports tensor_parallel and "
                f"replicate sharding; got {strategy}"
            )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Flux2FeedForward]:
        if self._sharding_strategy is None:
            raise ValueError(
                "Flux2FeedForward cannot be sharded without a sharding "
                "strategy."
            )
        devices_list = list(devices)
        sharded_in = self.linear_in.shard(devices_list)
        sharded_out = self.linear_out.shard(devices_list)
        shards: list[Flux2FeedForward] = []
        for device, lin_in, lin_out in zip(
            devices_list, sharded_in, sharded_out, strict=True
        ):
            sharded = Flux2FeedForward(
                dim=self.dim,
                dim_out=self.dim_out,
                inner_dim=self.inner_dim,
                bias=self.has_bias,
                dtype=self.dtype,
                devices=[device],
                quant_config=self.quant_config,
            )
            sharded.linear_in = lin_in
            sharded.linear_out = lin_out
            sharded._sharding_strategy = self._sharding_strategy
            shards.append(sharded)
        return shards


class Flux2PosEmbed(Module):
    def __init__(self, theta: int, axes_dim: tuple[int, ...]) -> None:
        """Initialize Flux2PosEmbed.

        Args:
            theta: Base frequency for RoPE
            axes_dim: Tuple of dimensions for each axis (e.g., (32, 32, 32, 32)).
        """
        super().__init__()
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def __call__(self, ids: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Compute rotary position embeddings.

        Args:
            ids: Position IDs of shape [S, len(axes_dim)].

        Returns:
            Tuple of (cos, sin) tensors of shape [S, sum(axes_dim)] for RoPE.
        """
        # Expected ids shape: [S, len(self.axes_dim)]
        cos_out = []
        sin_out = []

        # Convert to float for frequency computation

        pos = (
            ops.cast(ids, DType.float32) if ids.dtype != DType.float32 else ids
        )
        # Loop over each axis dimension

        for i, axis_dim in enumerate(self.axes_dim):
            cos, sin = get_1d_rotary_pos_embed(
                axis_dim,
                pos[..., i],
                theta=self.theta,
                use_real=True,
                repeat_interleave_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        # Concatenate all axes
        freqs_cos = ops.concat(cos_out, axis=-1)
        freqs_sin = ops.concat(sin_out, axis=-1)

        return freqs_cos, freqs_sin


class Flux2Attention(Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        *,
        dtype: DType,
        device: DeviceRef,
        quant: Flux2BlockQuant = Flux2BlockQuant(),
    ) -> None:
        """Initialize Flux2Attention.

        Args:
            query_dim: Dimension of query vectors.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout rate (not currently used).
            bias: Whether to use bias in Q/K/V projections.
            added_kv_proj_dim: If provided, enables dual-stream mode with separate encoder projections.
            added_proj_bias: Whether to use bias in encoder projections.
            out_bias: Whether to use bias in output projection.
            eps: Epsilon for RMSNorm.
            out_dim: Output dimension (defaults to query_dim).
            dtype: Weight dtype.
            device: Weight device.
            quant: Per-Linear quant plan; defaults to all-BF16.
        """
        super().__init__()
        del dropout
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        out_dim = out_dim if out_dim is not None else query_dim

        # Main Q/K/V projections
        self.to_q = Linear(
            query_dim,
            self.inner_dim,
            dtype,
            device,
            has_bias=bias,
            quant_config=quant.attn_qkv,
        )
        self.to_k = Linear(
            query_dim,
            self.inner_dim,
            dtype,
            device,
            has_bias=bias,
            quant_config=quant.attn_qkv,
        )
        self.to_v = Linear(
            query_dim,
            self.inner_dim,
            dtype,
            device,
            has_bias=bias,
            quant_config=quant.attn_qkv,
        )
        # QK normalization
        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
        # Output projection (skip dropout as it's not supported)
        self.to_out = LayerList(
            [
                Linear(
                    self.inner_dim,
                    out_dim,
                    dtype,
                    device,
                    has_bias=out_bias,
                    quant_config=quant.attn_out,
                )
            ]
        )

        # Optional: encoder projections
        self.norm_added_q: RMSNorm | None
        self.norm_added_k: RMSNorm | None
        self.add_q_proj: Linear | None
        self.add_k_proj: Linear | None
        self.add_v_proj: Linear | None
        self.to_add_out: Linear | None

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
            proj_bias = False if added_proj_bias is None else added_proj_bias
            self.add_q_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                device,
                has_bias=proj_bias,
                quant_config=quant.added_attn_qkv,
            )
            self.add_k_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                device,
                has_bias=proj_bias,
                quant_config=quant.added_attn_qkv,
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                dtype,
                device,
                has_bias=proj_bias,
                quant_config=quant.added_attn_qkv,
            )
            self.to_add_out = Linear(
                self.inner_dim,
                query_dim,
                dtype,
                device,
                has_bias=out_bias,
                quant_config=quant.added_attn_out,
            )
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None = None,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        """Apply dual-stream attention.

        Args:
            hidden_states: Image tokens of shape [B, S_img, D].
            encoder_hidden_states: Optional text tokens of shape [B, S_txt, D_enc]. If provided, enables dual-stream mode.
            image_rotary_emb: Optional tuple of (cos, sin) RoPE embeddings.

        Returns:
            If encoder_hidden_states is None: Output tensor of shape [B, S_img, out_dim].
            If encoder_hidden_states is provided: Tuple of (hidden_out, encoder_out) with shapes [B, S_img, out_dim] and [B, S_txt, query_dim].
        """
        batch_size = hidden_states.shape[0]
        # Project to Q/K/V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        seq_len = query.shape[1]

        # Reshape for multi-head attention: [B, S, D] -> [B, S, heads, dim_head]
        query = ops.reshape(
            query, [batch_size, seq_len, self.heads, self.head_dim]
        )
        key = ops.reshape(key, [batch_size, seq_len, self.heads, self.head_dim])
        value = ops.reshape(
            value, [batch_size, seq_len, self.heads, self.head_dim]
        )

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Handle encoder hidden states if provided
        if (
            encoder_hidden_states is not None
            and self.added_kv_proj_dim is not None
        ):
            if (
                self.add_q_proj is None
                or self.add_k_proj is None
                or self.add_v_proj is None
            ):
                raise ValueError("Encoder projections are not initialized")
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)
            encoder_seq_len = encoder_query.shape[1]
            # Reshape
            encoder_query = ops.reshape(
                encoder_query,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_key = ops.reshape(
                encoder_key,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_value = ops.reshape(
                encoder_value,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )

            # Apply normalization
            if self.norm_added_q is None or self.norm_added_k is None:
                raise ValueError("Encoder normalizations not initialized")
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = ops.concat([encoder_query, query], axis=1)
            key = ops.concat([encoder_key, key], axis=1)
            value = ops.concat([encoder_value, value], axis=1)

        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_flux2_qk_rope(query, key, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim**0.5)
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # hidden_states = F.flatten(hidden_states, 2, 3)
        # Reshape from [B, S, num_heads, head_dim] to [B, S, num_heads * head_dim]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = ops.reshape(
            hidden_states,
            [batch_size, seq_len, self.inner_dim],
        )
        hidden_states = ops.cast(hidden_states, query.dtype)

        # Split encoder and image outputs if dual-stream
        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            # Use slicing instead of F.split to handle symbolic dimensions
            encoder_out = hidden_states[:, :encoder_seq_len, :]
            hidden_out = hidden_states[:, encoder_seq_len:, :]

            # Project outputs
            hidden_out = self.to_out[0](hidden_out)
            if self.to_add_out is None:
                raise ValueError("Encoder output projection is not initialized")
            encoder_out = self.to_add_out(encoder_out)

            return hidden_out, encoder_out
        else:
            # Single stream output
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states


class Flux2ParallelSelfAttention(Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        *,
        dtype: DType,
        device: DeviceRef,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Initialize Flux2ParallelSelfAttention.

        Args:
            query_dim: Input dimension.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout rate (not used).
            bias: Whether to use bias in projections.
            out_bias: Whether to use bias in output projection.
            eps: Epsilon for RMSNorm.
            out_dim: Output dimension (defaults to query_dim).
            mlp_ratio: Multiplier for MLP hidden dimension.
            mlp_mult_factor: Multiplier for MLP projection (2 for SwiGLU).
            dtype: Weight dtype.
            device: Weight device.
        """
        super().__init__()
        del dropout
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        out_dim = out_dim if out_dim is not None else query_dim

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV + MLP input projection
        fused_dim = self.inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(
            query_dim,
            fused_dim,
            dtype,
            device,
            has_bias=bias,
            quant_config=quant_config,
        )

        # MLP activation
        self.mlp_act_fn = Flux2SwiGLU()

        # QK normalization
        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)

        # Fused output projection (Attention output + MLP output)
        self.to_out = Linear(
            self.inner_dim + self.mlp_hidden_dim,
            out_dim,
            dtype,
            device,
            has_bias=out_bias,
            quant_config=quant_config,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
    ) -> TensorValue:
        """Apply parallel self-attention and MLP.

        Args:
            hidden_states: Input tensor of shape [B, S, D].
            attention_mask: Optional attention mask (not used).
            image_rotary_emb: Optional tuple of (cos, sin) RoPE embeddings.

        Returns:
            Output tensor of shape [B, S, D].
        """
        # Fused projection
        fused = self.to_qkv_mlp_proj(hidden_states)

        # Split into QKV and MLP parts
        qkv_dim = self.inner_dim * 3
        mlp_dim = self.mlp_hidden_dim * self.mlp_mult_factor
        qkv, mlp_hidden_states = ops.split(fused, [qkv_dim, mlp_dim], axis=-1)

        # Split QKV
        query, key, value = ops.chunk(qkv, 3, axis=-1)

        # Reshape for multi-head: [B, S, D] -> [B, S, heads, dim_head]
        query = ops.reshape(
            query,
            [query.shape[0], query.shape[1], self.heads, self.head_dim],
        )
        key = ops.reshape(
            key,
            [key.shape[0], key.shape[1], self.heads, self.head_dim],
        )
        value = ops.reshape(
            value,
            [value.shape[0], value.shape[1], self.heads, self.head_dim],
        )

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_flux2_qk_rope(query, key, cos, sin)
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / (self.head_dim**0.5),
        )
        # hidden_states = F.flatten(hidden_states, 2, 3)
        # Reshape from [B, S, num_heads, head_dim] to [B, S, num_heads * head_dim]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = ops.reshape(
            hidden_states, [batch_size, seq_len, self.inner_dim]
        )
        hidden_states = ops.cast(hidden_states, query.dtype)
        # Process MLP stream
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        # Concatenate attention and MLP outputs
        hidden_states = ops.concat([hidden_states, mlp_hidden_states], axis=-1)

        # Final output projection
        output = self.to_out(hidden_states)

        return output
