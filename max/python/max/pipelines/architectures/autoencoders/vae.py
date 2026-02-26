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

from dataclasses import dataclass
from typing import Optional

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.nn.module_v3 import Conv2d, GroupNorm, Module, ModuleList

from .layers import Downsample2D, ResnetBlock2D, Upsample2D, VAEAttention


class DownEncoderBlock2D(Module[[Tensor], Tensor]):
    """Downsampling encoder block for 2D VAE.

    This module consists of multiple ResNet blocks followed by an optional
    downsampling layer. It progressively decreases spatial resolution while
    processing features through residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize DownEncoderBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout: Dropout rate (currently unused).
            num_layers: Number of ResNet blocks in this encoder block.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_time_scale_shift: Time embedding scale/shift mode (not used in encoder, temb=None).
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            resnet_pre_norm: Whether to apply normalization before ResNet.
            output_scale_factor: Scaling factor for output (currently unused).
            add_downsample: Whether to add downsampling layer after ResNet blocks.
            downsample_padding: Padding for the downsampling layer.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        resnets_list = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            if resnet_time_scale_shift == "spatial":
                raise NotImplementedError(
                    "resnet_time_scale_shift='spatial' is not supported in Max encoder. "
                    "Encoder uses temb=None, so only 'default' is supported."
                )

            resnet = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=out_channels,
                temb_channels=None,
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

        self.resnets = ModuleList(resnets_list)

        self.downsamplers: ModuleList[Downsample2D] | None = None
        if add_downsample:
            downsampler = Downsample2D(
                channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                name="op",
                kernel_size=3,
                norm_type=None,
                bias=True,
                device=device,
                dtype=dtype,
            )
            self.downsamplers = ModuleList([downsampler])

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply DownEncoderBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C_in, H, W].

        Returns:
            Output tensor of shape [N, C_out, H//2, W//2] (if downsampling) or
            [N, C_out, H, W] (if no downsampling).
        """
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, None)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)

        return hidden_states


class UpDecoderBlock2D(Module[[Tensor, Tensor | None], Tensor]):
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
        self.resnets = ModuleList(resnets_list)

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
            self.upsamplers: ModuleList[Upsample2D] | None = ModuleList(
                [upsampler]
            )
        else:
            self.upsamplers = None

    def forward(
        self, hidden_states: Tensor, temb: Tensor | None = None
    ) -> Tensor:
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


class MidBlock2D(Module[[Tensor, Tensor | None], Tensor]):
    """Middle block for 2D VAE.

    This module processes features at the middle of the VAE architecture,
    applying ResNet blocks with optional spatial attention mechanisms.
    It maintains spatial dimensions while processing features through
    residual connections and self-attention.
    """

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
        """Initialize MidBlock2D module.

        Args:
            in_channels: Number of input channels.
            temb_channels: Number of time embedding channels (None if not used).
            dropout: Dropout rate (currently unused).
            num_layers: Number of ResNet/attention layer pairs.
            resnet_eps: Epsilon value for ResNet GroupNorm layers.
            resnet_time_scale_shift: Time embedding scale/shift mode.
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            resnet_pre_norm: Whether to apply normalization before ResNet.
            add_attention: Whether to add attention layers between ResNet blocks.
            attention_head_dim: Dimension of each attention head.
            output_scale_factor: Scaling factor for output (currently unused).
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        resnets_list = []
        attentions_list: list[VAEAttention | None] = []

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

        self.resnets = ModuleList(resnets_list)

        if attentions_list:
            non_none_attentions = [
                attn for attn in attentions_list if attn is not None
            ]
            if non_none_attentions:
                self.attentions: ModuleList[VAEAttention] | None = ModuleList(
                    non_none_attentions
                )
                self.attention_indices = {
                    i
                    for i, attn in enumerate(attentions_list)
                    if attn is not None
                }
            else:
                self.attentions = None
                self.attention_indices = set()
        else:
            self.attentions = None
            self.attention_indices = set()

    def forward(
        self, hidden_states: Tensor, temb: Tensor | None = None
    ) -> Tensor:
        """Apply MidBlock2D forward pass.

        Args:
            hidden_states: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor.

        Returns:
            Output tensor of shape [N, C, H, W] with same spatial dimensions.
        """
        hidden_states = self.resnets[0](hidden_states, temb)

        attention_idx = 0
        for i in range(len(self.resnets) - 1):
            if self.attentions is not None and i in self.attention_indices:
                hidden_states = self.attentions[attention_idx](hidden_states)
                attention_idx += 1
            hidden_states = self.resnets[i + 1](hidden_states, temb)

        return hidden_states


@dataclass
class DecoderOutput:
    r"""Output of decoding method.

    Args:
        sample (`Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: Tensor
    commit_loss: Tensor | None = None


class Encoder(Module[[Tensor], Tensor]):
    r"""The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    This module progressively downsamples the input through multiple encoder blocks,
    applies a middle block for feature processing, and outputs encoded latents.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        down_block_types: The types of down blocks to use. Currently only supports "DownEncoderBlock2D".
        block_out_channels: The number of output channels for each block.
        layers_per_block: The number of layers per block.
        norm_num_groups: The number of groups for normalization.
        act_fn: The activation function to use (e.g., "silu").
        double_z: Whether to double the number of output channels for the last block.
        mid_block_add_attention: Whether to add attention in the middle block.
        device: Device reference for module placement.
        dtype: Data type for module parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
        use_quant_conv: bool = False,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize Encoder module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            down_block_types: Tuple of down block types (currently only "DownEncoderBlock2D").
            block_out_channels: Tuple of block output channels.
            layers_per_block: Number of layers per block.
            norm_num_groups: Number of groups for normalization.
            act_fn: Activation function name (e.g., "silu").
            double_z: Whether to double output channels for the last block.
            mid_block_add_attention: Whether to add attention in the middle block.
            use_quant_conv: Whether to add 1x1 conv after conv_out (encoder output -> latent moments).
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.down_blocks = ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type != "DownEncoderBlock2D":
                raise ValueError(
                    f"Unsupported down_block_type: {down_block_type}. "
                    "Currently only 'DownEncoderBlock2D' is supported."
                )

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=0.0,
                num_layers=self.layers_per_block,
                resnet_eps=1e-6,
                resnet_time_scale_shift="default",
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_pre_norm=True,
                output_scale_factor=1.0,
                add_downsample=not is_final_block,
                downsample_padding=0,
                device=device,
                dtype=dtype,
            )
            self.down_blocks.append(down_block)

        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_time_scale_shift="default",
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            resnet_pre_norm=True,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            output_scale_factor=1.0,
            device=device,
            dtype=dtype,
        )

        self.conv_norm_out = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[-1],
            eps=1e-6,
            affine=True,
        )

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[-1],
            out_channels=conv_out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.quant_conv: Conv2d | None = None
        if use_quant_conv:
            self.quant_conv = Conv2d(
                kernel_size=1,
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=True,
                device=device,
                permute=True,
            )

    def forward(self, sample: Tensor) -> Tensor:
        r"""The forward method of the `Encoder` class.

        Args:
            sample: Input tensor of shape [N, C_in, H, W].

        Returns:
            Output tensor of shape [N, C_out, H_latent, W_latent] (downsampled).
        """
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample, None)

        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        if self.quant_conv is not None:
            sample = self.quant_conv(sample)

        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the encoder model.

        Returns:
            Tuple of TensorType specifications for encoder input.
        """
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
        image_type = TensorType(
            self.dtype,
            shape=[
                "batch_size",
                self.in_channels,
                "image_height",
                "image_width",
            ],
            device=self.device,
        )

        return (image_type,)


class Decoder(Module[[Tensor, Tensor | None], Tensor]):
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

        self.post_quant_conv: Conv2d | None = None
        if use_post_quant_conv:
            self.post_quant_conv = Conv2d(
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

        self.conv_in = Conv2d(
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

        self.up_blocks = ModuleList(up_blocks_list)

        if norm_type == "spatial":
            raise NotImplementedError("SpatialNorm not implemented in MAX VAE")
        else:
            self.conv_norm_out = GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_out_channels[0],
                eps=1e-6,
                affine=True,
            )

        self.conv_out = Conv2d(
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

    def forward(self, z: Tensor, temb: Tensor | None = None) -> Tensor:
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
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the decoder model.

        Returns:
            Tuple of TensorType specifications for decoder input.
        """
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
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


class DiagonalGaussianDistribution:
    r"""Represents a diagonal Gaussian distribution for VAE latent space.

    This class represents a multivariate Gaussian distribution with diagonal
    covariance matrix, commonly used in Variational Autoencoders (VAEs) to
    model the latent space. It provides methods for sampling, computing KL
    divergence, and extracting the mode (mean) of the distribution.

    Args:
        parameters: Tensor of shape [N, 2*C, H, W] containing mean and logvar
            concatenated along the channel dimension. The first C channels are
            the mean, and the last C channels are the log variance.
        deterministic: If True, the distribution is deterministic (std=0).
            Defaults to False.
    """

    def __init__(self, parameters: Tensor, deterministic: bool = False) -> None:
        """Initialize DiagonalGaussianDistribution.

        Args:
            parameters: Tensor of shape [N, 2*C, H, W] containing mean and logvar.
            deterministic: Whether the distribution is deterministic.
        """
        self.parameters = parameters

        chunks = F.chunk(parameters, 2, axis=1)
        self.mean = chunks[0]
        self.logvar = chunks[1]

        self.logvar = F.min(F.max(self.logvar, -30.0), 20.0)

        self.deterministic = deterministic

        self.std = F.exp(0.5 * self.logvar)
        self.var = F.exp(self.logvar)

        if self.deterministic:
            self.var = Tensor.zeros_like(self.mean)
            self.std = Tensor.zeros_like(self.mean)

    def sample(self, generator: object | None = None) -> Tensor:
        """Sample from the distribution using reparameterization trick.

        Generates a random sample from the distribution by sampling from a
        standard normal distribution and transforming it using the mean and
        standard deviation.

        Args:
            generator: Random number generator (currently unused in Max,
                kept for compatibility with diffusers API).

        Returns:
            Sampled tensor of shape [N, C, H, W] with same shape as mean.
        """
        sample = random.normal(
            shape=self.mean.shape,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )

        x = self.mean + F.mul(self.std, sample)
        return x

    def kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> Tensor:
        """Compute KL divergence with another distribution or standard normal.

        Computes the Kullback-Leibler divergence between this distribution and
        either a standard normal distribution (if other is None) or another
        DiagonalGaussianDistribution.

        Args:
            other: Optional other DiagonalGaussianDistribution to compute KL
                divergence with. If None, computes KL divergence with standard
                normal distribution.

        Returns:
            Tensor containing KL divergence values.
        """
        if self.deterministic:
            return Tensor.constant([0.0], dtype=DType.float32)

        if other is None:
            kl_term = F.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            kl_term = F.sum(kl_term, axis=3)
            kl_term = F.sum(kl_term, axis=2)
            kl_term = F.sum(kl_term, axis=1)
            return 0.5 * kl_term
        else:
            kl_term = (
                F.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )
            kl_term = F.sum(kl_term, axis=3)
            kl_term = F.sum(kl_term, axis=2)
            kl_term = F.sum(kl_term, axis=1)
            return 0.5 * kl_term

    def nll(self, sample: Tensor, dims: tuple[int, ...] = (1, 2, 3)) -> Tensor:
        """Compute negative log-likelihood of a sample.

        Computes the negative log-likelihood of a given sample under this
        distribution.

        Args:
            sample: Sample tensor to compute NLL for.
            dims: Dimensions to sum over. Defaults to (1, 2, 3) for spatial dims.

        Returns:
            Tensor containing negative log-likelihood values.
        """
        if self.deterministic:
            return Tensor.constant([0.0], dtype=DType.float32)

        logtwopi = np.log(2.0 * np.pi)
        nll_term = (
            logtwopi + self.logvar + F.pow(sample - self.mean, 2) / self.var
        )

        sorted_dims = sorted(dims, reverse=True)
        for dim in sorted_dims:
            nll_term = F.sum(nll_term, axis=dim)

        return 0.5 * nll_term

    def mode(self) -> Tensor:
        """Return the mode (mean) of the distribution.

        For a Gaussian distribution, the mode is equal to the mean.

        Returns:
            Mean tensor of shape [N, C, H, W].
        """
        return self.mean
