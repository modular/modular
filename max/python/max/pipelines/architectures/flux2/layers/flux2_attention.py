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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.module_v3 import Linear, Module, module_dataclass
from max.nn.module_v3.sequential import ModuleList

flash_attention_gpu = F.functional(_flash_attention_gpu)

from max.nn.module_v3.norm import RMSNorm

from .embeddings import apply_rotary_emb, get_1d_rotary_pos_embed


@module_dataclass
class Flux2SwiGLU(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Output tensor of shape [..., dim//2].
        """
        x1, x2 = F.chunk(x, chunks=2, axis=-1)
        return F.silu(x1) * x2


class Flux2FeedForward(Module[[Tensor], Tensor]):
    linear_in: Linear
    act_fn: Flux2SwiGLU
    linear_out: Linear

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
    ):
        """Initialize Flux2FeedForward.

        Args:
            dim: Input dimension.
            dim_out: Output dimension (defaults to dim).
            mult: Multiplier for hidden dimension (defaults to 3.0).
            inner_dim: Explicit inner dimension (overrides mult if provided).
            bias: Whether to use bias in linear layers.
        """
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Flux2SwiGLU will reduce the dimension by half
        self.linear_in = Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
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


class Flux2PosEmbed(Module[[Tensor], tuple[Tensor, Tensor]]):
    theta: int
    axes_dim: tuple[int, ...]

    def __init__(self, theta: int, axes_dim: tuple[int, ...]):
        """Initialize Flux2PosEmbed.

        Args:
            theta: Base frequency for RoPE
            axes_dim: Tuple of dimensions for each axis (e.g., (32, 32, 32, 32)).
        """
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
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
        pos = ids.cast(DType.float32) if ids.dtype != DType.float32 else ids

        # Loop over each axis dimension
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                use_real=True,
                repeat_interleave_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        # Concatenate all axes
        freqs_cos = F.concat(cos_out, axis=-1)
        freqs_sin = F.concat(sin_out, axis=-1)

        return freqs_cos, freqs_sin


class Flux2Attention(Module[..., Tensor | tuple[Tensor, Tensor]]):
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
    ):
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
        """
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        out_dim = out_dim if out_dim is not None else query_dim

        # Main Q/K/V projections
        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(query_dim, self.inner_dim, bias=bias)

        # QK normalization
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Output projection (skip dropout as it's not supported)
        self.to_out = ModuleList()
        self.to_out.append(Linear(self.inner_dim, out_dim, bias=out_bias))

        # Optional: encoder projections
        self.norm_added_q: RMSNorm | None
        self.norm_added_k: RMSNorm | None
        self.add_q_proj: Linear | None
        self.add_k_proj: Linear | None
        self.add_v_proj: Linear | None
        self.to_add_out: Linear | None
        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.add_q_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias if added_proj_bias is not None else False,
            )
            self.add_k_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias if added_proj_bias is not None else False,
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias if added_proj_bias is not None else False,
            )
            self.to_add_out = Linear(self.inner_dim, query_dim, bias=out_bias)
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        # attention_mask: Optional[Tensor] = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
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
        query = F.reshape(
            query, [batch_size, seq_len, self.heads, self.head_dim]
        )
        key = F.reshape(key, [batch_size, seq_len, self.heads, self.head_dim])
        value = F.reshape(
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
                raise ValueError("Encoder projections not initialized")
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)
            encoder_seq_len = encoder_query.shape[1]
            # Reshape
            encoder_query = F.reshape(
                encoder_query,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_key = F.reshape(
                encoder_key,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_value = F.reshape(
                encoder_value,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )

            # Apply normalization
            if self.norm_added_q is None or self.norm_added_k is None:
                raise ValueError("Encoder normalizations not initialized")
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Concatenate encoder and image tokens
            query = F.concat([encoder_query, query], axis=1)
            key = F.concat([encoder_key, key], axis=1)
            value = F.concat([encoder_value, value], axis=1)

        # Apply rotary embeddings if provided
        # Store original dtype to cast back after RoPE (which may upcast to float32)
        original_dtype = query.dtype
        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                query,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-1,
                sequence_dim=1,
            )
            key = apply_rotary_emb(
                key,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-1,
                sequence_dim=1,
            )
            # Cast back to original dtype to match value
            query = query.cast(original_dtype)
            key = key.cast(original_dtype)

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
        hidden_states = F.reshape(
            hidden_states, [batch_size, seq_len, self.inner_dim]
        )
        hidden_states = hidden_states.cast(query.dtype)

        # Split encoder and image outputs if dual-stream
        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            # Use slicing instead of F.split to handle symbolic dimensions
            encoder_out = hidden_states[:, :encoder_seq_len, :]
            hidden_out = hidden_states[:, encoder_seq_len:, :]

            # Project outputs
            hidden_out = self.to_out[0](hidden_out)
            if self.to_add_out is None:
                raise ValueError("Encoder output projection not initialized")
            encoder_out = self.to_add_out(encoder_out)

            return hidden_out, encoder_out
        else:
            # Single stream output
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states


class Flux2ParallelSelfAttention(Module[[Tensor], Tensor]):
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
    ):
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
        """
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        out_dim = out_dim if out_dim is not None else query_dim

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV + MLP input projection
        fused_dim = self.inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(query_dim, fused_dim, bias=bias)

        # MLP activation
        self.mlp_act_fn = Flux2SwiGLU()

        # QK normalization
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Fused output projection (Attention output + MLP output)
        self.to_out = Linear(
            self.inner_dim + self.mlp_hidden_dim, out_dim, bias=out_bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
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
        qkv, mlp_hidden_states = F.split(fused, [qkv_dim, mlp_dim], axis=-1)

        # Split QKV
        query, key, value = F.chunk(qkv, 3, axis=-1)

        # Reshape for multi-head: [B, S, D] -> [B, S, heads, dim_head]
        query = F.reshape(
            query, [query.shape[0], query.shape[1], self.heads, self.head_dim]
        )
        key = F.reshape(
            key, [key.shape[0], key.shape[1], self.heads, self.head_dim]
        )
        value = F.reshape(
            value, [value.shape[0], value.shape[1], self.heads, self.head_dim]
        )

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply rotary embeddings
        # Store original dtype to cast back after RoPE (which may upcast to float32)
        original_dtype = query.dtype
        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                query,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-1,
                sequence_dim=1,
            )
            key = apply_rotary_emb(
                key,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-1,
                sequence_dim=1,
            )
            # Cast back to original dtype to match value
            query = query.cast(original_dtype)
            key = key.cast(original_dtype)

        # Attention computation
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
        hidden_states = F.reshape(
            hidden_states, [batch_size, seq_len, self.inner_dim]
        )
        hidden_states = hidden_states.cast(query.dtype)

        # Process MLP stream
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)  # type: ignore[arg-type]

        # Concatenate attention and MLP outputs
        hidden_states = F.concat([hidden_states, mlp_hidden_states], axis=-1)

        # Final output projection
        output = self.to_out(hidden_states)

        return output
