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

import math
from math import prod

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.conv import Conv3D
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .layers.conv import CausalConv1d
from .layers.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from .layers.motion_block import MotionConv2d, MotionResBlock
from .layers.normalization import WanLayerNorm
from .layers.transformer import WanTransformerBlock
from .model_config import WanConfigBase


class WanTextProjection(Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.linear_1 = Linear(
            in_dim=in_features,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.linear_2 = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, caption: TensorValue) -> TensorValue:
        hidden_states = self.linear_1(caption)
        hidden_states = ops.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanImageEmbedder(Module):
    """Image embedding for Wan 2.1 I2V: LayerNorm → GEGLU FFN → LayerNorm.

    Matches diffusers' FeedForward(image_dim, dim, mult=1, activation_fn="gelu")
    with pre/post norms.  Weight keys::

        image_embedder.norm1.{weight,bias}
        image_embedder.ff.net.0.proj.{weight,bias}   (GEGLU gate+value)
        image_embedder.ff.net.2.{weight,bias}         (output linear)
        image_embedder.norm2.{weight,bias}
    """

    def __init__(
        self,
        image_dim: int,
        out_dim: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        # Matches diffusers FeedForward(image_dim, out_dim, mult=1, activation_fn="gelu"):
        #   norm1(image_dim) → Linear(image_dim→image_dim) → GELU →
        #   Linear(image_dim→out_dim) → norm2(out_dim)
        self.norm1 = WanLayerNorm(
            image_dim,
            elementwise_affine=True,
            use_bias=True,
            dtype=dtype,
            device=device,
        )
        self.ff_proj = Linear(
            in_dim=image_dim,
            out_dim=image_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.ff_out = Linear(
            in_dim=image_dim,
            out_dim=out_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.norm2 = WanLayerNorm(
            out_dim,
            elementwise_affine=True,
            use_bias=True,
            dtype=dtype,
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.norm1(x)
        x = ops.gelu(self.ff_proj(x))
        x = self.ff_out(x)
        return self.norm2(x)


class WanTimeTextImageEmbedding(Module):
    def __init__(
        self,
        dim: int,
        freq_dim: int,
        text_dim: int,
        num_layers: int,
        *,
        image_dim: int | None = None,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=dim,
            dtype=dtype,
            device=device,
        )
        # Projects SiLU(temb) to 6 modulation params per block
        self.time_proj = Linear(
            in_dim=dim,
            out_dim=dim * 6,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.text_embedder = WanTextProjection(
            in_features=text_dim,
            hidden_size=dim,
            dtype=dtype,
            device=device,
        )
        # Optional image embedder (Wan 2.1 I2V)
        self.image_embedder: WanImageEmbedder | None = None
        if image_dim is not None:
            self.image_embedder = WanImageEmbedder(
                image_dim=image_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
            )

    def __call__(
        self, timestep: TensorValue, encoder_hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        # Sinusoidal timestep embedding (computed in float32 for precision).
        # Cast to the model's working dtype (bf16) for the MLP, matching
        # diffusers' behavior: float32 embedding → cast to weight dtype → MLP.
        timesteps_emb = self.timesteps_proj(timestep)  # [B, freq_dim] float32
        timesteps_emb = ops.cast(
            timesteps_emb, encoder_hidden_states.dtype
        )  # → bf16
        temb = self.time_embedder(timesteps_emb)  # [B, dim]

        # Timestep projection for modulation: SiLU then linear
        timestep_proj = self.time_proj(ops.silu(temb))  # [B, dim*6]
        # Reshape to [B, 6, dim] for per-block modulation
        timestep_proj = ops.reshape(
            timestep_proj,
            [timestep_proj.shape[0], 6, timestep_proj.shape[1] // 6],
        )

        # Text projection
        text_emb = self.text_embedder(encoder_hidden_states)  # [B, S, dim]

        return temb, timestep_proj, text_emb


class WanTransformerPreProcess(Module):
    """Patch embedding + condition embedding (compiled separately).

    When ``is_animate=True``, this also injects pose latents and returns
    image embeddings for cross-attention.
    """

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
        is_animate: bool = False,
    ) -> None:
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.is_animate = is_animate

        self.patch_embedding = Conv3D(
            depth=config.patch_size[0],
            height=config.patch_size[1],
            width=config.patch_size[2],
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.pose_patch_embedding: Conv3D | None = None
        if is_animate:
            self.pose_patch_embedding = Conv3D(
                depth=config.patch_size[0],
                height=config.patch_size[1],
                width=config.patch_size[2],
                in_channels=config.latent_channels,
                out_channels=dim,
                stride=config.patch_size,
                dtype=dtype,
                device=device,
                has_bias=True,
                permute=True,
            )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
            image_dim=config.image_dim if is_animate else None,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        timestep: TensorValue,
        encoder_hidden_states: TensorValue,
        clip_features: TensorValue | None = None,
        pose_hidden_states: TensorValue | None = None,
    ) -> (
        tuple[TensorValue, TensorValue, TensorValue, TensorValue]
        | tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]
    ):
        batch_size = hidden_states.shape[0]
        hs = ops.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = ops.permute(hs, [0, 4, 1, 2, 3])

        if self.is_animate:
            if self.pose_patch_embedding is None or pose_hidden_states is None:
                raise ValueError(
                    "pose_hidden_states is required when is_animate=True"
                )
            # Pose patch embedding [B, 16, T, H, W]
            pose_embed = self.pose_patch_embedding(pose_hidden_states)
            # Add pose to frames 1+ (skip frame 0 = reference image)
            hs_ref = hs[:, :, :1, :, :]
            hs_rest = hs[:, :, 1:, :, :]
            hs_rest = ops.rebind(
                hs_rest,
                shape=[
                    hs_rest.shape[0],
                    hs_rest.shape[1],
                    pose_embed.shape[2],
                    hs_rest.shape[3],
                    hs_rest.shape[4],
                ],
            )
            hs_rest = hs_rest + pose_embed
            hs = ops.concat([hs_ref, hs_rest], axis=2)

        seq_len = hs.shape[2] * hs.shape[3] * hs.shape[4]
        hs = ops.reshape(hs, [batch_size, self.inner_dim, seq_len])
        hs = ops.permute(hs, [0, 2, 1])

        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )
        if not self.is_animate:
            return hs, temb, timestep_proj, text_emb

        if (
            clip_features is None
            or self.condition_embedder.image_embedder is None
        ):
            raise ValueError("clip_features is required when is_animate=True")
        image_embeds = self.condition_embedder.image_embedder(clip_features)
        return hs, temb, timestep_proj, text_emb, image_embeds


class WanTransformerPostProcess(Module):
    """Output modulation + unpatchify (compiled separately)."""

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.scale_shift_table = Weight(
            "scale_shift_table", dtype, [1, 2, dim], device
        )
        self.norm_out = WanLayerNorm(
            dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.proj_out = Linear(
            in_dim=dim,
            out_dim=config.out_channels * prod(config.patch_size),
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        temb: TensorValue,
        spatial_shape: TensorValue,
    ) -> TensorValue:
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.patch_size
        ppf = spatial_shape.shape[0]
        pph = spatial_shape.shape[1]
        ppw = spatial_shape.shape[2]

        mod = self.scale_shift_table + ops.reshape(
            temb, [batch_size, 1, self.inner_dim]
        )
        shift = mod[:, :1, :]
        scale = mod[:, 1:, :]
        hs = self.norm_out(hidden_states) * (1.0 + scale) + shift
        hs = self.proj_out(hs)
        hs = ops.rebind(
            hs,
            shape=[
                batch_size,
                ppf * pph * ppw,
                self.out_channels * p_t * p_h * p_w,
            ],
        )

        hs = ops.reshape(
            hs,
            [batch_size, ppf, pph, ppw, p_t, p_h, p_w, self.out_channels],
        )
        hs = ops.permute(hs, [0, 7, 1, 4, 2, 5, 3, 6])
        hs = ops.reshape(
            hs,
            [batch_size, self.out_channels, ppf * p_t, pph * p_h, ppw * p_w],
        )
        return ops.cast(hs, DType.bfloat16)


class WanTransformer3DModel(Module):
    """Full transformer (for reference / single-graph compilation)."""

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.patch_embedding = Conv3D(
            depth=config.patch_size[0],
            height=config.patch_size[1],
            width=config.patch_size[2],
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
            image_dim=getattr(config, "image_dim", None),
            dtype=dtype,
            device=device,
        )
        self.blocks = LayerList(
            [
                WanTransformerBlock(
                    dim=dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    text_dim=dim,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.scale_shift_table = Weight(
            "scale_shift_table", dtype, [1, 2, dim], device
        )
        self.norm_out = WanLayerNorm(
            dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.proj_out = Linear(
            in_dim=dim,
            out_dim=config.out_channels * prod(config.patch_size),
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        timestep: TensorValue,
        encoder_hidden_states: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
    ) -> TensorValue:
        batch_size = hidden_states.shape[0]
        orig_T = hidden_states.shape[2]
        orig_H = hidden_states.shape[3]
        orig_W = hidden_states.shape[4]
        p_t, p_h, p_w = self.patch_size
        ppf = orig_T // p_t
        pph = orig_H // p_h
        ppw = orig_W // p_w

        hs = ops.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = ops.permute(hs, [0, 4, 1, 2, 3])
        hs = ops.reshape(hs, [batch_size, self.inner_dim, ppf * pph * ppw])
        hs = ops.permute(hs, [0, 2, 1])

        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        # Rebind RoPE to match the sequence length derived from spatial dims.
        seq_len = ppf * pph * ppw
        rope_cos = ops.rebind(rope_cos, shape=[seq_len, self.head_dim])
        rope_sin = ops.rebind(rope_sin, shape=[seq_len, self.head_dim])

        for block in self.blocks:
            hs = block(hs, text_emb, timestep_proj, rope_cos, rope_sin)

        mod = self.scale_shift_table + ops.reshape(
            temb, [batch_size, 1, self.inner_dim]
        )
        shift = mod[:, :1, :]
        scale = mod[:, 1:, :]
        hs = self.norm_out(hs) * (1.0 + scale) + shift
        hs = self.proj_out(hs)

        hs = ops.reshape(
            hs,
            [batch_size, ppf, pph, ppw, p_t, p_h, p_w, self.out_channels],
        )
        hs = ops.permute(hs, [0, 7, 1, 4, 2, 5, 3, 6])
        hs = ops.reshape(
            hs,
            [batch_size, self.out_channels, ppf * p_t, pph * p_h, ppw * p_w],
        )
        return ops.cast(hs, self.config.dtype)


class WanAnimateTransformer3DModel(Module):
    """Wan-Animate transformer with module-based face adapter injection."""

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        self.inject_interval = config.inject_face_latents_blocks

        self.pre = WanTransformerPreProcess(
            config, is_animate=True, dtype=dtype, device=device
        )
        self.blocks = LayerList(
            [
                WanTransformerBlock(
                    dim=dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    text_dim=dim,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    added_kv_proj_dim=config.added_kv_proj_dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.num_layers)
            ]
        )
        num_face_adapters = config.num_layers // self.inject_interval
        self.face_adapter = LayerList(
            [
                WanAnimateFaceBlock(
                    dim=dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    eps=config.eps,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_face_adapters)
            ]
        )
        self.post = WanTransformerPostProcess(
            config, dtype=dtype, device=device
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        timestep: TensorValue,
        encoder_hidden_states: TensorValue,
        clip_features: TensorValue,
        pose_hidden_states: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
        spatial_shape: TensorValue,
        face_emb: TensorValue,
        num_temporal_frames: TensorValue,
    ) -> TensorValue:
        hs, temb, timestep_proj, text_emb, image_embeds = self.pre(
            hidden_states,
            timestep,
            encoder_hidden_states,
            clip_features,
            pose_hidden_states,
        )

        # Tie RoPE sequence dim to the actual token sequence produced by pre.
        seq_len = hs.shape[1]
        rope_cos = ops.rebind(
            rope_cos, shape=[seq_len, self.config.attention_head_dim]
        )
        rope_sin = ops.rebind(
            rope_sin, shape=[seq_len, self.config.attention_head_dim]
        )

        adapter_idx = 0
        for i in range(len(self.blocks)):
            hs = self.blocks[i](
                hs,
                text_emb,
                timestep_proj,
                rope_cos,
                rope_sin,
                image_embeds,
            )

            if i % self.inject_interval == 0 and adapter_idx < len(
                self.face_adapter
            ):
                adapter_out = self.face_adapter[adapter_idx](
                    hs, face_emb, num_temporal_frames
                )
                adapter_out = ops.rebind(
                    adapter_out,
                    shape=[hs.shape[0], hs.shape[1], hs.shape[2]],
                )
                hs = hs + adapter_out
                adapter_idx += 1

        return self.post(hs, temb, spatial_shape)


class WanAnimateFaceEncoder(Module):
    """Face encoder: motion vectors -> face embeddings via CausalConv1d.

    Architecture:
        conv1_local(512 -> 4096, k=3) -> reshape to 4 heads ->
        norm1 -> SiLU ->
        conv2(1024 -> 1024, k=3, s=2) -> norm2 -> SiLU ->
        conv3(1024 -> 1024, k=3, s=2) -> norm3 -> SiLU ->
        out_proj(1024 -> 5120) + padding_tokens
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 1024,
        out_dim: int = 5120,
        num_heads: int = 4,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # CausalConv1d weights: [out_channels, in_channels, kernel_size]
        # conv1_local expands to num_heads * hidden_dim
        self.conv1_local = CausalConv1d(
            in_dim,
            hidden_dim * num_heads,
            kernel_size=3,
            stride=1,
            dtype=dtype,
            device=device,
        )
        self.norm1 = WanLayerNorm(
            hidden_dim,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )

        self.conv2 = CausalConv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=2,
            dtype=dtype,
            device=device,
        )
        self.norm2 = WanLayerNorm(
            hidden_dim,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )

        self.conv3 = CausalConv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=2,
            dtype=dtype,
            device=device,
        )
        self.norm3 = WanLayerNorm(
            hidden_dim,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )

        self.out_proj = Linear(
            in_dim=hidden_dim,
            out_dim=out_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        # Learned null token: [1, 1, 1, out_dim]
        self.padding_tokens = Weight(
            "padding_tokens",
            dtype,
            [1, 1, 1, out_dim],
            device,
        )

    def __call__(self, motion_vectors: TensorValue) -> TensorValue:
        """Encode motion vectors to face embeddings.

        Args:
            motion_vectors: [B, T, 512] motion vectors from motion encoder.

        Returns:
            [B, T//4 + 1, num_heads + 1, out_dim] face embeddings
            (with prepended zero frame and padding token channel).
        """
        batch_size = motion_vectors.shape[0]
        t_in = motion_vectors.shape[1]

        # conv1_local: [B, T, 512] -> [B, T, 4096]
        x = self.conv1_local(motion_vectors)

        # Reshape to multi-head: [B, T, 4*1024] -> [B, T, 4, 1024] -> [B, 4, T, 1024] -> [B*4, T, 1024]
        x = ops.reshape(x, [batch_size, t_in, self.num_heads, self.hidden_dim])
        x = ops.permute(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size * self.num_heads, t_in, self.hidden_dim])

        # norm1 -> SiLU
        x = self.norm1(x)
        x = ops.silu(x)

        # conv2: stride-2 downsample T -> T//2
        x = self.conv2(x)
        x = self.norm2(x)
        x = ops.silu(x)

        # conv3: stride-2 downsample T//2 -> T//4
        x = self.conv3(x)
        x = self.norm3(x)
        x = ops.silu(x)

        # t_out = T//4
        t_out = x.shape[1]

        # out_proj: [B*4, T//4, 1024] -> [B*4, T//4, 5120]
        x = self.out_proj(x)

        out_dim = x.shape[2]

        # Reshape back: [B*4, T//4, 5120] -> [B, 4, T//4, 5120] -> [B, T//4, 4, 5120]
        x = ops.reshape(x, [batch_size, self.num_heads, t_out, out_dim])
        x = ops.permute(x, [0, 2, 1, 3])

        # Add padding token channel: [B, T//4, 4, D] -> [B, T//4, 5, D]
        padding = ops.broadcast_to(
            self.padding_tokens,
            [batch_size, t_out, 1, out_dim],
        )
        x = ops.concat([x, padding], axis=2)

        # Prepend zero frame: [B, 1, 5, D] of zeros
        zero_frame = x[:, :1, :, :] - x[:, :1, :, :]
        x = ops.concat([zero_frame, x], axis=1)

        return x  # [B, T//4 + 1, 5, 5120]


class WanAnimateFaceBlock(Module):
    """Face adapter cross-attention block.

    Temporally-aligned attention: video tokens attend to face motion tokens.
    Video [B, S, D] reshaped to [B*T, S/T, H, head_dim] attends to
    face [B, T, 5, D] reshaped to [B*T, 5, H, head_dim].
    """

    def __init__(
        self,
        dim: int = 5120,
        num_heads: int = 40,
        head_dim: int = 128,
        eps: float = 1e-6,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_k = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_v = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_out = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.norm_no_affine = WanLayerNorm(
            dim,
            eps=1e-5,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.norm_q = RMSNorm(head_dim, dtype=dtype, eps=eps)
        self.norm_k = RMSNorm(head_dim, dtype=dtype, eps=eps)

    def __call__(
        self,
        hidden_states: TensorValue,
        face_emb: TensorValue,
        num_temporal_frames: TensorValue,
    ) -> TensorValue:
        """Apply face adapter cross-attention.

        Args:
            hidden_states: [B, S, D] video features (S = T * H_p * W_p).
            face_emb: [B, T, 5, D] face motion embeddings.
            num_temporal_frames: [1] int32 tensor with T value (for rebind).

        Returns:
            [B, S, D] residual output to add to hidden_states.
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        t_face = face_emb.shape[1]
        face_tokens = face_emb.shape[2]  # 5

        # LayerNorm (elementwise_affine=False) on video features
        x = self.norm_no_affine(hidden_states)

        # Q from video features: [B, S, D]
        q = self.to_q(x)

        # LayerNorm (elementwise_affine=False) on face motion
        # face_emb: [B, T, 5, D] — flatten for norm, then restore
        face_flat_for_norm = ops.reshape(
            face_emb,
            [batch_size * t_face * face_tokens, self.dim],
        )
        face_normed = self.norm_no_affine(
            ops.reshape(
                face_flat_for_norm,
                [batch_size * t_face, face_tokens, self.dim],
            )
        )

        # K, V from face motion
        face_kv_flat = ops.reshape(
            face_normed,
            [batch_size * t_face * face_tokens, self.dim],
        )
        k_flat = self.to_k(face_kv_flat)
        v_flat = self.to_v(face_kv_flat)

        # Reshape K, V: -> [B*T, 5, H, head_dim]
        k = ops.reshape(
            k_flat,
            [batch_size * t_face, face_tokens, self.num_heads, self.head_dim],
        )
        v = ops.reshape(
            v_flat,
            [batch_size * t_face, face_tokens, self.num_heads, self.head_dim],
        )

        # Reshape Q: [B, S, D] -> [B*T, S/T, H, head_dim]
        # rebind Q to assert seq_len == t_face * spatial_per_frame
        spatial_per_frame = seq_len // t_face
        q = ops.rebind(
            q,
            shape=[
                batch_size,
                t_face * spatial_per_frame,
                self.dim,
            ],
        )
        q_4d = ops.reshape(
            q,
            [
                batch_size * t_face,
                spatial_per_frame,
                self.num_heads,
                self.head_dim,
            ],
        )

        # QK-norm
        q_4d = self.norm_q(q_4d)
        k = self.norm_k(k)

        # Flash attention
        original_dtype = q_4d.dtype
        scale = 1.0 / (self.head_dim**0.5)
        attn_out = flash_attention_gpu(
            q_4d,
            k,
            v,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # Reshape back: [B*T, S/T, H, head_dim] -> [B, S, D]
        attn_out = ops.reshape(
            attn_out, [batch_size, t_face * spatial_per_frame, self.dim]
        )
        attn_out = ops.cast(attn_out, original_dtype)

        return self.to_out(attn_out)


# Channel sizes for the StyleGAN2-based appearance encoder.
WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES: dict[int, int] = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
}


class WanAnimateMotionEncoder(Module):
    """MAX-native motion encoder replacing the PyTorch bridge.

    StyleGAN2-based CNN: face crops ``[B, 3, 512, 512]`` → motion vectors
    ``[B, 512]``.

    Weights must be pre-processed before loading:

    - Conv weights: ``OIHW → RSCF``, scale ``1/sqrt(fan_in)`` baked in.
    - Linear weights: ``[out, in] → [in, out]``, scale baked in.
    - ``motion_synthesis_weight`` replaced by pre-computed ``q_matrix``
      from QR decomposition.
    - Indexed keys follow LayerList naming: ``res_blocks.{i}.`` and
      ``motion_network.{i}.``.
    """

    def __init__(
        self,
        size: int = 512,
        style_dim: int = 512,
        motion_dim: int = 20,
        out_dim: int = 512,
        motion_blocks: int = 5,
        channels: dict[int, int] | None = None,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        if channels is None:
            channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        # conv_in: Conv2d(3 → channels[size], k=1) + activation
        init_ch = channels[size]  # 32
        self.conv_in = MotionConv2d(
            3,
            init_ch,
            1,
            stride=1,
            padding=0,
            has_blur=False,
            has_activation=True,
            dtype=dtype,
            device=device,
        )

        # Res blocks: progressive spatial downsampling 512 → 4
        log_size = int(math.log(size, 2))  # 9
        res_blocks: list[MotionResBlock] = []
        in_ch = init_ch
        for i in range(log_size, 2, -1):
            out_ch = channels[2 ** (i - 1)]
            res_blocks.append(
                MotionResBlock(in_ch, out_ch, dtype=dtype, device=device)
            )
            in_ch = out_ch
        self.res_blocks = LayerList(res_blocks)

        # conv_out: Conv2d(512 → style_dim, k=4), no bias, no activation
        self.conv_out = MotionConv2d(
            in_ch,
            style_dim,
            4,
            stride=1,
            padding=0,
            has_blur=False,
            has_activation=False,
            dtype=dtype,
            device=device,
        )

        # Motion network: linear layers (no intermediate activations)
        motion_layers: list[Linear] = [
            Linear(
                in_dim=style_dim,
                out_dim=style_dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            for _ in range(motion_blocks - 1)
        ]
        motion_layers.append(
            Linear(
                in_dim=style_dim,
                out_dim=motion_dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
        )
        self.motion_network = LayerList(motion_layers)

        # Q matrix from QR decomposition (pre-computed, float32 for precision)
        self.q_matrix = Weight(
            "q_matrix", DType.float32, [out_dim, motion_dim], device
        )

    def __call__(self, face_image: TensorValue) -> TensorValue:
        """Encode face crops to motion vectors.

        Args:
            face_image: ``[B, 3, 512, 512]`` in ``[-1, 1]`` (NCHW).

        Returns:
            ``[B, 512]`` motion vectors.
        """
        # NCHW → NHWC
        x = ops.permute(face_image, [0, 2, 3, 1])

        # Appearance encoding through conv layers
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.conv_out(x)  # [B, 1, 1, style_dim]

        # Squeeze spatial dims → [B, style_dim]
        x = ops.reshape(x, [x.shape[0], x.shape[3]])

        # Motion feature extraction through linear layers
        for layer in self.motion_network:
            x = layer(x)
        # x: [B, motion_dim=20]

        # Motion synthesis via Linear Motion Decomposition:
        #   diag_embed(x) @ Q.T, summed over dim=1, simplifies to x @ Q.T.
        x = ops.cast(x, DType.float32)
        q_t = ops.permute(self.q_matrix, [1, 0])  # [motion_dim, out_dim]
        motion_vec = ops.matmul(x, q_t)  # [B, out_dim=512]
        return ops.cast(motion_vec, DType.bfloat16)
