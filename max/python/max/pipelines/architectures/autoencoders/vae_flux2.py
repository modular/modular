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

"""Graph API VAE encoder components for Flux2.

Ports the ModuleV2-based encoder from autoencoders_modulev3/ to the Graph API
(max.nn.layer.Module) so that the ImageEncoder CompiledComponent can be built
without max.experimental dependencies.
"""

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.conv import Conv2d
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.group_norm import GroupNorm


class ResnetBlock2D(Module):
    """Residual block for 2D VAE encoder/decoder.

    Two convolutions with group normalization and SiLU activation,
    plus an optional shortcut projection when channel counts differ.
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            device=device or DeviceRef.GPU(),
        )

        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype or DType.bfloat16,
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
            device=device or DeviceRef.GPU(),
        )

        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype or DType.bfloat16,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        self.conv_shortcut: Conv2d | None = None
        if self.use_conv_shortcut or in_channels != out_channels:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype or DType.bfloat16,
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
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        h = ops.silu(self.norm1(x))
        h = self.conv1(h)
        h = ops.silu(self.norm2(h))
        h = self.conv2(h)
        return h + shortcut


class Downsample2D(Module):
    """2D downsampling layer with strided convolution.

    Reduces spatial dimensions by half using a stride-2 convolution,
    with explicit padding when padding=0.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        norm_type: str | None = None,
        eps: float | None = None,
        elementwise_affine: bool | None = None,
        bias: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        if norm_type is not None:
            raise NotImplementedError(
                f"norm_type={norm_type!r} is not supported in Graph API Downsample2D"
            )

        self.conv: Conv2d | None = None
        if use_conv:
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=self.channels,
                out_channels=self.out_channels,
                dtype=dtype or DType.bfloat16,
                stride=2,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )
        elif self.channels != self.out_channels:
            raise ValueError(
                f"When use_conv=False, channels must equal out_channels. "
                f"Got channels={self.channels}, out_channels={self.out_channels}"
            )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        if self.use_conv and self.padding == 0:
            paddings = [0, 0, 0, 0, 0, 1, 0, 1]
            hidden_states = ops.pad(
                hidden_states, paddings=paddings, mode="constant", value=0
            )

        if self.use_conv:
            assert self.conv is not None
            hidden_states = self.conv(hidden_states)
        else:
            raise NotImplementedError(
                "avg_pool2d downsampling is not supported in Graph API Downsample2D"
            )

        return hidden_states


class VAEAttention(Module):
    """Spatial self-attention for VAE models.

    Converts [N, C, H, W] to sequence format, applies multi-head
    flash attention, and converts back with a residual connection.
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
            device=device or DeviceRef.GPU(),
        )

        _dtype = dtype or DType.bfloat16
        _device = device or DeviceRef.GPU()

        self.to_q = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=_dtype,
            device=_device,
            has_bias=True,
        )
        self.to_k = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=_dtype,
            device=_device,
            has_bias=True,
        )
        self.to_v = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=_dtype,
            device=_device,
            has_bias=True,
        )
        self.to_out = LayerList(
            [
                Linear(
                    in_dim=self.inner_dim,
                    out_dim=query_dim,
                    dtype=_dtype,
                    device=_device,
                    has_bias=True,
                )
            ]
        )

        self.scale = 1.0 / math.sqrt(dim_head)

    def __call__(self, x: TensorValue) -> TensorValue:
        residual = x

        x = self.group_norm(x)

        n, c, h, w = x.shape
        seq_len = h * w

        x = ops.reshape(x, [n, c, seq_len])
        x = ops.permute(x, [0, 2, 1])

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # flash_attention_gpu expects [batch, seq, heads, dim_head]
        q = ops.reshape(q, [n, seq_len, self.heads, self.dim_head])
        k = ops.reshape(k, [n, seq_len, self.heads, self.dim_head])
        v = ops.reshape(v, [n, seq_len, self.heads, self.dim_head])

        out = flash_attention_gpu(
            q, k, v, mask_variant=MHAMaskVariant.NULL_MASK, scale=self.scale
        )

        out = ops.reshape(out, [n, seq_len, self.inner_dim])
        out = self.to_out[0](out)

        out = ops.permute(out, [0, 2, 1])
        out = ops.reshape(out, [n, c, h, w])

        return residual + out


class DownEncoderBlock2D(Module):
    """Downsampling encoder block with ResNet layers and optional spatial downsampling."""

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
        resnets_list = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            if resnet_time_scale_shift == "spatial":
                raise NotImplementedError(
                    "resnet_time_scale_shift='spatial' is not supported. "
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
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, None)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)

        return hidden_states


class MidBlock2D(Module):
    """Middle block with ResNet layers and optional spatial attention."""

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
                self.attention_indices: set[int] = {
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
        hidden_states = self.resnets[0](hidden_states, temb)

        attention_idx = 0
        for i in range(len(self.resnets) - 1):
            if self.attentions is not None and i in self.attention_indices:
                hidden_states = self.attentions[attention_idx](hidden_states)
                attention_idx += 1
            hidden_states = self.resnets[i + 1](hidden_states, temb)

        return hidden_states


class Encoder(Module):
    """VAE encoder that maps images to latent representations.

    Progressively downsamples through encoder blocks, applies a middle
    block with optional attention, and outputs latent moments.
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
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype

        self.conv_in = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            dtype=dtype or DType.bfloat16,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
        )

        down_blocks_list = []
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
            device=device or DeviceRef.GPU(),
        )

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = Conv2d(
            kernel_size=3,
            in_channels=block_out_channels[-1],
            out_channels=conv_out_channels,
            dtype=dtype or DType.bfloat16,
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
                dtype=dtype or DType.bfloat16,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
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
        sample = ops.silu(sample)
        sample = self.conv_out(sample)

        if self.quant_conv is not None:
            sample = self.quant_conv(sample)

        return sample

    def input_types(self) -> tuple[TensorType, ...]:
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
