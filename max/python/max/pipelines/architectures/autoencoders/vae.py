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

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue
from max.nn.activation import activation_function_from_name
from max.nn.conv import Conv2d
from max.nn.layer import LayerList, Module
from max.nn.norm import GroupNorm

from .layers import Downsample2D, ResnetBlock2D, Upsample2D, VAEAttention


class DownEncoderBlock2D(Module):
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
            resnet_time_scale_shift: Time embedding scale/shift mode (not used
                in encoder, temb=None).
            resnet_act_fn: Activation function for ResNet blocks.
            resnet_groups: Number of groups for ResNet GroupNorm.
            resnet_pre_norm: Whether to apply normalization before ResNet.
            output_scale_factor: Scaling factor for output (currently unused).
            add_downsample: Whether to add downsampling layer after ResNet
                blocks.
            downsample_padding: Padding for the downsampling layer.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        del dropout, resnet_pre_norm, output_scale_factor
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

        self.resnets = LayerList(resnets_list)
        self.downsamplers: LayerList | None = None
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
            self.downsamplers = LayerList([downsampler])

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
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


class UpDecoderBlock2D(Module):
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
            self.upsamplers: LayerList | None = LayerList([upsampler])
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
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class MidBlock2D(Module):
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
            add_attention: Whether to add attention layers between ResNet
                blocks.
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

        self.resnets = LayerList(resnets_list)

        if attentions_list:
            non_none_attentions = [
                attn for attn in attentions_list if attn is not None
            ]
            if non_none_attentions:
                self.attentions: LayerList | None = LayerList(
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
        sample (`TensorValue` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: TensorValue
    commit_loss: TensorValue | None = None


class Encoder(Module):
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
        if dtype is None:
            raise ValueError("dtype must be set for Encoder")
        if device is None:
            raise ValueError("device must be set for Encoder")
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype
        self.activation = activation_function_from_name(act_fn)
        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        output_channel = block_out_channels[0]
        down_blocks_list = []
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
            down_blocks_list.append(down_block)

        self.down_blocks = LayerList(down_blocks_list)

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
            device=device,
        )
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[-1],
            out_channels=conv_out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
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
                has_bias=True,
                device=device,
                permute=True,
            )

    def __call__(self, sample: TensorValue) -> TensorValue:
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
        sample = self.activation(sample)
        sample = self.conv_out(sample)
        if self.quant_conv is not None:
            sample = self.quant_conv(sample)
        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the encoder model.

        Returns:
            Tuple of TensorType specifications for encoder input.
        """
        return (
            TensorType(
                self.dtype,
                shape=[
                    "batch_size",
                    self.in_channels,
                    "image_height",
                    "image_width",
                ],
                device=self.device,
            ),
        )


class Decoder(Module):
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
        if dtype is None:
            raise ValueError("dtype must be set for Decoder")
        if device is None:
            raise ValueError("device must be set for Decoder")

        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype
        self.activation = activation_function_from_name(act_fn)

        self.post_quant_conv: Conv2d | None = None
        if use_post_quant_conv:
            self.post_quant_conv = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=in_channels,
                dtype=dtype,
                stride=1,
                padding=0,
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
            has_bias=True,
            device=device,
            permute=True,
        )
        temb_channels = in_channels if norm_type == "spatial" else None
        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=temb_channels,
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
            )
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
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
        sample = (
            self.post_quant_conv(z) if self.post_quant_conv is not None else z
        )
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, temb)
        for up_block in self.up_blocks:
            sample = up_block(sample, temb)
        sample = self.conv_norm_out(sample)
        sample = self.activation(sample)
        sample = self.conv_out(sample)
        return sample

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the decoder model.

        Returns:
            Tuple of TensorType specifications for decoder input.
        """
        return (
            TensorType(
                self.dtype,
                shape=[
                    "batch_size",
                    self.in_channels,
                    "latent_height",
                    "latent_width",
                ],
                device=self.device,
            ),
        )


class DiagonalGaussianDistribution:
    r"""Represents a diagonal Gaussian distribution for VAE latent space.

    This wrapper intentionally stays lightweight for the Buffer-based VAE path.
    """

    def __init__(
        self,
        mean: object,
        moments: object | None = None,
    ) -> None:
        """Initialize DiagonalGaussianDistribution.

        Args:
            mean: Mean tensor or mode tensor.
            moments: Optional raw moments tensor containing additional VAE
                distribution parameters.
        """
        self.mean = mean
        self.parameters = moments

    def sample(self, generator: object | None = None) -> object:
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
        del generator
        return self.mean

    def mode(self) -> object:
        """Return the mode (mean) of the distribution."""
        return self.mean
