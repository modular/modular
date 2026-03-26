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
        super().__init__()
        del dropout, resnet_pre_norm, output_scale_factor
        if resnet_time_scale_shift == "spatial":
            raise NotImplementedError(
                "resnet_time_scale_shift='spatial' is not supported in Max encoder."
            )
        resnets = []
        for idx in range(num_layers):
            input_channels = in_channels if idx == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
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
            )
        self.resnets = LayerList(resnets)
        self.downsamplers: LayerList | None = None
        if add_downsample:
            self.downsamplers = LayerList(
                [
                    Downsample2D(
                        channels=out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        kernel_size=3,
                        norm_type=None,
                        bias=True,
                        device=device,
                        dtype=dtype,
                    )
                ]
            )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, None)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
        return hidden_states


class UpDecoderBlock2D(Module):
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
        super().__init__()
        del resolution_idx, dropout, resnet_time_scale_shift
        del resnet_pre_norm, output_scale_factor, temb_channels
        resnets = []
        for idx in range(num_layers):
            input_channels = in_channels if idx == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
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
            )
        self.resnets = LayerList(resnets)
        self.upsamplers: LayerList | None = None
        if add_upsample:
            self.upsamplers = LayerList(
                [
                    Upsample2D(
                        channels=out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        interpolate=True,
                        device=device,
                        dtype=dtype,
                    )
                ]
            )

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class MidBlock2D(Module):
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
        super().__init__()
        del dropout, resnet_time_scale_shift, resnet_pre_norm
        del output_scale_factor
        resnets = [
            ResnetBlock2D(
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
        ]
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                VAEAttention(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    num_groups=resnet_groups,
                    eps=resnet_eps,
                    device=device,
                    dtype=dtype,
                )
                if add_attention
                else None
            )
            resnets.append(
                ResnetBlock2D(
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
            )
        self.resnets = LayerList(resnets)
        self.attentions = LayerList(
            [attn for attn in attentions if attn is not None]
        )
        self.attention_indices = {
            idx for idx, attn in enumerate(attentions) if attn is not None
        }

    def __call__(
        self, hidden_states: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        hidden_states = self.resnets[0](hidden_states, temb)
        attention_idx = 0
        for idx in range(len(self.resnets) - 1):
            if idx in self.attention_indices:
                hidden_states = self.attentions[attention_idx](hidden_states)
                attention_idx += 1
            hidden_states = self.resnets[idx + 1](hidden_states, temb)
        return hidden_states


@dataclass
class DecoderOutput:
    sample: TensorValue
    commit_loss: TensorValue | None = None


class Encoder(Module):
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
        super().__init__()
        if dtype is None:
            raise ValueError("dtype must be set for Encoder")
        if device is None:
            raise ValueError("device must be set for Encoder")
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
        down_blocks = []
        for idx, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[idx]
            is_final_block = idx == len(block_out_channels) - 1
            if down_block_type != "DownEncoderBlock2D":
                raise ValueError(
                    f"Unsupported down_block_type: {down_block_type}."
                )
            down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    downsample_padding=0,
                    device=device,
                    dtype=dtype,
                )
            )
        self.down_blocks = LayerList(down_blocks)

        self.mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
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
        super().__init__()
        if dtype is None:
            raise ValueError("dtype must be set for Decoder")
        if device is None:
            raise ValueError("device must be set for Decoder")

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
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            device=device,
            dtype=dtype,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for idx, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[idx]
            is_final_block = idx == len(block_out_channels) - 1
            if up_block_type != "UpDecoderBlock2D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")
            up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    resolution_idx=idx,
                    num_layers=layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final_block,
                    temb_channels=temb_channels,
                    device=device,
                    dtype=dtype,
                )
            )
        self.up_blocks = LayerList(up_blocks)

        if norm_type == "spatial":
            raise NotImplementedError("SpatialNorm not implemented in MAX VAE")
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
    """Runtime wrapper used by the Flux2 image-to-image path."""

    def __init__(
        self,
        mean: object,
        moments: object | None = None,
    ) -> None:
        self.mean = mean
        self.parameters = moments

    def sample(self, generator: object | None = None) -> object:
        del generator
        return self.mean

    def mode(self) -> object:
        return self.mean
