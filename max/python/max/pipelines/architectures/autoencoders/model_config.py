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

from typing import Any, ClassVar

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from pydantic import Field


class AutoencoderKLConfigBase(MAXModelConfigBase):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: list[str] = Field(default_factory=list, max_length=4)
    up_block_types: list[str] = Field(default_factory=list, max_length=4)
    block_out_channels: list[int] = Field(default_factory=list, max_length=4)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    shift_factor: float | None = None
    latents_mean: tuple[float] | None = None
    latents_std: tuple[float] | None = None
    force_upcast: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLConfig(AutoencoderKLConfigBase):
    config_name: ClassVar[str] = "config.json"

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": encoding.dtype,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLConfig(**init_dict)

class AutoencoderKLLTX2VideoConfigBase(MAXModelConfigBase):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    decoder_block_out_channels: tuple[int, ...] = (256, 512, 1024)
    decoder_layers_per_block: tuple[int, ...] = (5, 5, 5, 5)
    decoder_spatio_temporal_scaling: tuple[bool, ...] = (True, True, True)
    decoder_inject_noise: tuple[bool, ...] = (False, False, False, False)
    upsample_residual: tuple[bool, ...] = (True, True, True)
    upsample_factor: tuple[int, ...] = (2, 2, 2)
    timestep_conditioning: bool = False
    patch_size: int = 4
    patch_size_t: int = 1
    resnet_norm_eps: float = 1e-6
    scaling_factor: float = 1.0
    decoder_causal: bool = True
    decoder_spatial_padding_mode: str = "reflect"
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLLTX2VideoConfig(AutoencoderKLLTX2VideoConfigBase):
    config_name: ClassVar[str] = "config.json"

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLLTX2VideoConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLLTX2VideoConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": encoding.dtype,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLLTX2VideoConfig(**init_dict)


class AutoencoderKLLTX2AudioConfigBase(MAXModelConfigBase):
    base_channels: int = 128
    output_channels: int = 2
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: list[int] | None = None
    in_channels: int = 2
    resolution: int = 256
    latent_channels: int = 8
    norm_type: str = "pixel"
    causality_axis: str | None = "height"
    dropout: float = 0.0
    mid_block_add_attention: bool = False
    sample_rate: int = 16000
    mel_hop_length: int = 160
    is_causal: bool = True
    mel_bins: int | None = 64
    double_z: bool = True
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLLTX2AudioConfig(AutoencoderKLLTX2AudioConfigBase):
    config_name: ClassVar[str] = "config.json"

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLLTX2AudioConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLLTX2AudioConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": encoding.dtype,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLLTX2AudioConfig(**init_dict)
