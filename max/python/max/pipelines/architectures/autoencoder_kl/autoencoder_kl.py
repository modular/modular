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

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass

import max.nn as nn
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn import GroupNorm
from max.nn.layer.layer_list import LayerList

from .layers import Upsample2D
from .model_config import AutoencoderKLConfig


class ResnetBlock2D(nn.Module):
    """Residual block for 2D VAE decoder.

    This module implements a residual block with two convolutional layers,
    group normalization, and optional shortcut connection. It supports
    time embedding conditioning and configurable activation functions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int | None,
        groups: int,
        groups_out: int,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        use_conv_shortcut: bool = False,
        conv_shortcut_bias: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize ResnetBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            temb_channels: Number of time embedding channels (None if not used).
            groups: Number of groups for first GroupNorm.
            groups_out: Number of groups for second GroupNorm.
            eps: Epsilon value for GroupNorm layers.
            non_linearity: Activation function name (e.g., "silu").
            use_conv_shortcut: Whether to use convolutional shortcut.
            conv_shortcut_bias: Whether to use bias in shortcut convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            device=device,
            dtype=dtype,
        )

        self.conv1 = nn.Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.norm2 = GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            device=device,
            dtype=dtype,
        )

        self.conv2 = nn.Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.conv_shortcut = None
        if self.use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )
        elif in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )

    def __call__(
        self, x: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply ResnetBlock2D forward pass.

        Args:
            x: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor (currently unused).

        Returns:
            Output tensor of shape [N, C_out, H, W] with residual connection.
        """
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        h = ops.silu(self.norm1(x))
        h = self.conv1(h)

        h = ops.silu(self.norm2(h))
        h = self.conv2(h)

        return h + shortcut


class UpDecoderBlock2D(nn.Module):
    """Upsampling decoder block for 2D VAE.

    This module consists of multiple ResNet blocks followed by an optional
    upsampling layer. It progressively increases spatial resolution while
    processing features through residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: int | None = None,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize UpDecoderBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            resolution_idx: Optional resolution index for tracking.
            dropout: Dropout rate (currently unused).
            num_layers: Number of ResNet blocks in this decoder block.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_time_scale_shift: Time embedding scale/shift mode.
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            resnet_pre_norm: Whether to apply normalization before ResNet.
            output_scale_factor: Scaling factor for output (currently unused).
            add_upsample: Whether to add upsampling layer after ResNet blocks.
            temb_channels: Number of time embedding channels (None if not used).
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        resnets_list = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnet = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                groups_out=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                use_conv_shortcut=False,
                conv_shortcut_bias=True,
                device=device,
                dtype=dtype,
            )
            resnets_list.append(resnet)
        self.resnets = LayerList(resnets_list)

        if add_upsample:
            upsampler = Upsample2D(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                name="conv",
                kernel_size=3,
                padding=1,
                bias=True,
                interpolate=True,
                device=device,
                dtype=dtype,
            )
            self.upsamplers = LayerList([upsampler])
        else:
            self.upsamplers = None

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply UpDecoderBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C_in, H, W].
            temb: Optional time embedding tensor.

        Returns:
            Output tensor of shape [N, C_out, H*2, W*2] (if upsampling) or
            [N, C_out, H, W] (if no upsampling).
        """
        # Process through all resnet blocks
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

        # Apply upsampling if configured (compile-time decision)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class VAEAttention(nn.Module):
    """Spatial attention module for VAE models.

    This module performs self-attention on 2D spatial features by:
    1. Converting [N, C, H, W] to [N, H*W, C] sequence format
    2. Applying scaled dot-product attention (optimized for small sequences)
    3. Converting back to [N, C, H, W] format

    Note: Manual attention is used instead of flash_attention_gpu because
    VAE attention typically has small sequence lengths (H*W) where flash
    attention overhead outweighs benefits.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        num_groups: int = 32,
        eps: float = 1e-6,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize VAE attention module.

        Args:
            query_dim: Dimension of query (number of channels).
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            num_groups: Number of groups for GroupNorm.
            eps: Epsilon value for GroupNorm.
            device: Device reference.
            dtype: Data type.
        """
        super().__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        self.group_norm = GroupNorm(
            num_groups=num_groups,
            num_channels=query_dim,
            eps=eps,
            affine=True,
            device=device,
            dtype=dtype,
        )

        self.to_q = nn.Linear(
            query_dim, self.inner_dim, has_bias=True, device=device, dtype=dtype
        )
        self.to_k = nn.Linear(
            query_dim, self.inner_dim, has_bias=True, device=device, dtype=dtype
        )
        self.to_v = nn.Linear(
            query_dim, self.inner_dim, has_bias=True, device=device, dtype=dtype
        )
        self.to_out = LayerList(
            [
                nn.Linear(
                    self.inner_dim,
                    query_dim,
                    has_bias=True,
                    device=device,
                    dtype=dtype,
                )
            ]
        )

        self.scale = 1.0 / math.sqrt(dim_head)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply spatial attention to 2D image tensor.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            Output tensor of shape [N, C, H, W] with residual connection.
        """
        residual = x

        x = self.group_norm(x)

        n, c, h, w = x.shape
        seq_len = h * w

        x = ops.reshape(x, (n, c, seq_len))
        x = ops.permute(x, (0, 2, 1))

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = ops.reshape(q, (n, seq_len, self.heads, self.dim_head))
        q = ops.permute(q, (0, 2, 1, 3))
        k = ops.reshape(k, (n, seq_len, self.heads, self.dim_head))
        k = ops.permute(k, (0, 2, 1, 3))
        v = ops.reshape(v, (n, seq_len, self.heads, self.dim_head))
        v = ops.permute(v, (0, 2, 1, 3))

        attn = q @ ops.permute(k, (0, 1, 3, 2)) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = attn @ v

        out = ops.permute(out, (0, 2, 1, 3))
        out = ops.reshape(out, (n, seq_len, self.inner_dim))

        out = self.to_out[0](out)

        out = ops.permute(out, (0, 2, 1))
        out = ops.reshape(out, (n, c, h, w))

        return residual + out


class MidBlock2D(nn.Module):
    """Internal MAX module for MidBlock2D graph generation."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int | None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize MidBlock2D module."""
        super().__init__()
        resnets_list = []
        attentions_list = []

        resnet = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            groups=resnet_groups,
            groups_out=resnet_groups,
            eps=resnet_eps,
            non_linearity=resnet_act_fn,
            use_conv_shortcut=False,
            conv_shortcut_bias=True,
            device=device,
            dtype=dtype,
        )
        resnets_list.append(resnet)

        for _i in range(num_layers):
            if add_attention:
                attn = VAEAttention(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    num_groups=resnet_groups,
                    eps=resnet_eps,
                    device=device,
                    dtype=dtype,
                )
                attentions_list.append(attn)
            else:
                attentions_list.append(None)

            resnet = ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                groups_out=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                use_conv_shortcut=False,
                conv_shortcut_bias=True,
                device=device,
                dtype=dtype,
            )
            resnets_list.append(resnet)

        self.resnets = LayerList(resnets_list)
        self.attentions = (
            LayerList(attentions_list) if attentions_list else None
        )

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply MidBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor.

        Returns:
            Output tensor of shape [N, C, H, W] with same spatial dimensions.
        """
        hidden_states = self.resnets[0](hidden_states, temb)

        for i in range(len(self.resnets) - 1):
            if self.attentions is not None and self.attentions[i] is not None:
                hidden_states = self.attentions[i](hidden_states)
            hidden_states = self.resnets[i + 1](hidden_states, temb)

        return hidden_states


@dataclass
class DecoderOutput:
    r"""Output of decoding method.

    Args:
        sample (`TensorValue` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: TensorValue
    commit_loss: TensorValue | None = None


class Decoder(nn.Module):
    """VAE decoder for generating images from latent representations.

    This decoder progressively upsamples latent features through multiple
    decoder blocks, applying ResNet layers and attention mechanisms to
    reconstruct high-resolution images from compressed latent codes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",
        mid_block_add_attention: bool = True,
        use_post_quant_conv: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize Decoder module.

        Args:
            in_channels: Number of input channels (latent channels).
            out_channels: Number of output channels (image channels).
            up_block_types: Tuple of upsampling block types.
            block_out_channels: Tuple of channel counts for each decoder block.
            layers_per_block: Number of ResNet layers per decoder block.
            norm_num_groups: Number of groups for GroupNorm layers.
            act_fn: Activation function name (e.g., "silu").
            norm_type: Normalization type ("group" or "spatial").
            mid_block_add_attention: Whether to add attention in middle block.
            use_post_quant_conv: Whether to use post-quantization convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.layers_per_block = layers_per_block
        self.session = None
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype

        self.post_quant_conv = None
        if use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=in_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=True,
                device=device,
                permute=True,
            )

        self.conv_in = nn.Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[-1],
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        temb_channels = in_channels if norm_type == "spatial" else None
        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_time_scale_shift=(
                "default" if norm_type == "group" else norm_type
            ),
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            resnet_pre_norm=True,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            output_scale_factor=1.0,
            device=device,
            dtype=dtype,
        )

        up_blocks_list = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "UpDecoderBlock2D":
                up_block = UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    resolution_idx=i,
                    dropout=0.0,
                    num_layers=self.layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_time_scale_shift=norm_type,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_pre_norm=True,
                    output_scale_factor=1.0,
                    add_upsample=not is_final_block,
                    temb_channels=temb_channels,
                    device=device,
                    dtype=dtype,
                )
                up_blocks_list.append(up_block)
            else:
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            prev_output_channel = output_channel

        self.up_blocks = LayerList(up_blocks_list)

        if norm_type == "spatial":
            raise NotImplementedError("SpatialNorm not implemented in MAX VAE")
        else:
            self.conv_norm_out = GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_out_channels[0],
                eps=1e-6,
                affine=True,
                device=device,
                dtype=dtype,
            )

        self.conv_out = nn.Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

    def __call__(
        self, z: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply Decoder forward pass.

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].
            temb: Optional time embedding tensor.

        Returns:
            Decoded image tensor of shape [N, C_out, H, W] where H and W are
            upsampled from H_latent and W_latent.
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        sample = self.conv_in(z)
        sample = self.mid_block(sample, temb)

        for up_block in self.up_blocks:
            sample = up_block(sample, temb)

        sample = self.conv_norm_out(sample)
        sample = ops.silu(sample)
        sample = self.conv_out(sample)

        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the decoder model.

        Returns:
            Tuple of TensorType specifications for decoder input.
        """
        latent_type = TensorType(
            self.dtype,
            shape=[
                "batch_size",
                self.in_channels,
                "latent_height",
                "latent_width",
            ],
            device=self.device,
        )

        return (latent_type,)


class AutoencoderKL(nn.Module):
    r"""A VAE model with KL loss for encoding images into latents and decoding latent representations into images."""

    def __init__(
        self,
        config: AutoencoderKLConfig,
    ):
        """Initialize VAE AutoencoderKL model.

        Args:
            config: Autoencoder configuration containing channel sizes, block
                structure, normalization settings, and device/dtype information.
        """
        super().__init__()
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            norm_type="group",
            mid_block_add_attention=config.mid_block_add_attention,
            use_post_quant_conv=config.use_post_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )

    def __call__(self, *args, **kwargs):
        pass
