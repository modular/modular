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

"""Z-Image DiT core model (Graph API / ModuleV2)."""

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .layers.attention import ZImageAttention
from .layers.embeddings import RopeEmbedder, TimestepEmbedder
from .model_config import ZImageConfig

ADALN_EMBED_DIM = 256


class LayerNorm(Module):
    """Layer normalisation with optional learned affine parameters."""

    weight: Weight | None
    bias: Weight | None

    def __init__(
        self,
        dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = Weight("weight", dtype, (dim,), device=device)
            self.bias = (
                Weight("bias", dtype, (dim,), device=device)
                if use_bias
                else None
            )
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.weight is None:
            gamma = ops.broadcast_to(
                ops.constant(1.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            gamma = self.weight

        if self.bias is None:
            beta = ops.broadcast_to(
                ops.constant(0.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            beta = self.bias

        return ops.layer_norm(x, gamma=gamma, beta=beta, epsilon=self.eps)


class FeedForward(Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.w1 = Linear(
            in_dim=dim,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.w2 = Linear(
            in_dim=hidden_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.w3 = Linear(
            in_dim=dim,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(Module):
    """Single transformer block with optional adaLN modulation."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        *,
        dtype: DType,
        device: DeviceRef,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        del n_kv_heads

        self.modulation = modulation
        self.dim = dim

        self.attention = ZImageAttention(
            dim=dim,
            n_heads=n_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=int(dim / 3 * 8),
            dtype=dtype,
            device=device,
        )
        self.attention_norm1 = RMSNorm(dim, dtype=dtype, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, dtype=dtype, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, dtype=dtype, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, dtype=dtype, eps=norm_eps)

        self.adaLN_modulation = (
            Linear(
                in_dim=min(dim, ADALN_EMBED_DIM),
                out_dim=4 * dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            if modulation
            else None
        )

    def __call__(
        self,
        x: TensorValue,
        freqs_cis: TensorValue,
        adaln_input: TensorValue | None = None,
    ) -> TensorValue:
        if self.modulation:
            if adaln_input is None:
                raise ValueError("adaln_input is required when modulation=True")
            if self.adaLN_modulation is None:
                raise ValueError("adaLN_modulation is not initialized")

            mod = ops.unsqueeze(self.adaLN_modulation(adaln_input), 1)
            d = self.dim
            scale_msa = 1.0 + mod[:, :, :d]
            gate_msa = ops.tanh(mod[:, :, d : 2 * d])
            scale_mlp = 1.0 + mod[:, :, 2 * d : 3 * d]
            gate_mlp = ops.tanh(mod[:, :, 3 * d :])

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            x = x + gate_mlp * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(Module):
    """Final projection layer with adaLN conditioning."""

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.norm_final = LayerNorm(
            hidden_size,
            dtype=dtype,
            device=device,
            eps=1e-6,
            elementwise_affine=False,
            use_bias=False,
        )
        self.linear = Linear(
            in_dim=hidden_size,
            out_dim=out_channels,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.adaLN_modulation = Linear(
            in_dim=min(hidden_size, ADALN_EMBED_DIM),
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue, c: TensorValue) -> TensorValue:
        scale = 1.0 + self.adaLN_modulation(ops.silu(c))
        x = self.norm_final(x) * ops.unsqueeze(scale, 1)
        return self.linear(x)


class ZImageTransformer2DModel(Module):
    """Z-Image diffusion transformer (DiT) model."""

    def __init__(self, config: ZImageConfig) -> None:
        super().__init__()

        dim = config.dim
        n_heads = config.n_heads
        norm_eps = config.norm_eps
        qk_norm = config.qk_norm
        cap_feat_dim = config.cap_feat_dim
        n_layers = config.n_layers
        n_refiner_layers = config.n_refiner_layers
        axes_dims = config.axes_dims
        rope_theta = config.rope_theta
        dtype = config.dtype
        device = config.device

        patch_size = config.all_patch_size[0]
        f_patch_size = config.all_f_patch_size[0]
        in_channels = (
            config.in_channels * patch_size * patch_size * f_patch_size
        )
        out_channels = in_channels

        self.packed_channels = in_channels
        self.max_dtype = dtype
        self.max_device = device
        self.cap_feat_dim = cap_feat_dim
        self.t_scale = config.t_scale
        self.axes_dims = axes_dims

        self.x_embedder = Linear(
            in_dim=in_channels,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.final_layer = FinalLayer(
            hidden_size=dim,
            out_channels=out_channels,
            dtype=dtype,
            device=device,
        )

        self.noise_refiner = LayerList(
            [
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=config.n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=True,
                )
                for _ in range(n_refiner_layers)
            ]
        )
        self.context_refiner = LayerList(
            [
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=config.n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=False,
                )
                for _ in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(
            out_size=min(dim, ADALN_EMBED_DIM),
            mid_size=1024,
            dtype=dtype,
            device=device,
        )
        self.cap_norm = RMSNorm(cap_feat_dim, dtype=dtype, eps=norm_eps)
        self.cap_proj = Linear(
            in_dim=cap_feat_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.layers = LayerList(
            [
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=config.n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    dtype=dtype,
                    device=device,
                    modulation=True,
                )
                for _ in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        if head_dim != sum(axes_dims):
            raise ValueError(
                f"head_dim ({head_dim}) must equal sum(axes_dims) ({sum(axes_dims)})"
            )

        self.rope_embedder = RopeEmbedder(
            theta=rope_theta,
            axes_dims=axes_dims,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.packed_channels],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", "text_seq_len", self.cap_feat_dim],
                device=self.max_device,
            ),
            TensorType(
                DType.float32,
                shape=["batch_size"],
                device=self.max_device,
            ),
            TensorType(
                DType.int64,
                shape=["image_seq_len", len(self.axes_dims)],
                device=self.max_device,
            ),
            TensorType(
                DType.int64,
                shape=["text_seq_len", len(self.axes_dims)],
                device=self.max_device,
            ),
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
    ) -> tuple[TensorValue]:
        x = self.x_embedder(hidden_states)
        t_emb = self.t_embedder(timestep * self.t_scale)
        t_emb = ops.cast(t_emb, x.dtype)

        cap = self.cap_proj(self.cap_norm(encoder_hidden_states))

        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        img_seq_len = img_ids.shape[0]
        unified_ids = ops.concat([img_ids, txt_ids], axis=0)
        unified_freqs = ops.cast(self.rope_embedder(unified_ids), x.dtype)
        img_freqs = unified_freqs[:img_seq_len]
        txt_freqs = unified_freqs[img_seq_len:]

        for block in self.noise_refiner:
            x = block(x, freqs_cis=img_freqs, adaln_input=t_emb)

        for block in self.context_refiner:
            cap = block(cap, freqs_cis=txt_freqs)

        img_len = x.shape[1]
        x = ops.concat([x, cap], axis=1)

        for block in self.layers:
            x = block(x, freqs_cis=unified_freqs, adaln_input=t_emb)

        x = x[:, :img_len, :]
        x = self.final_layer(x, t_emb)
        return (x,)
