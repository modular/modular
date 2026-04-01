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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
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
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLConfig(AutoencoderKLConfigBase):
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
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )

        # Z-Image/Flux pipelines use decode-time inverse scaling:
        # latents = (latents / scaling_factor) + shift_factor.
        # Keep config deterministic by normalizing missing/None shift_factor.
        if init_dict.get("shift_factor") is None:
            init_dict["shift_factor"] = 0.0

        # Guard against invalid decode-time divide-by-zero.
        if (
            "scaling_factor" in init_dict
            and float(init_dict["scaling_factor"]) == 0.0
        ):
            raise ValueError("`scaling_factor` must be non-zero.")

        return AutoencoderKLConfig(**init_dict)


class AutoencoderKLWanConfigBase(MAXModelConfigBase):
    # Defaults mirror Wan2.2 AutoencoderKLWan config.
    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    latents_mean: tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class AutoencoderKLWanConfig(AutoencoderKLWanConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLWanConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLWanConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLWanConfig(**init_dict)


class AutoencoderKLQwenImageConfigBase(MAXModelConfigBase):
    """Configuration for the QwenImage 3D causal VAE (Wan-2.1 based)."""

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: list[int] = Field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_scales: list[float] = Field(default_factory=list)
    temporal_downsample: list[bool] = Field(
        default_factory=lambda: [False, True, True]
    )
    dropout: float = 0.0
    latents_mean: list[float] = Field(default_factory=list)
    latents_std: list[float] = Field(default_factory=list)
    device: DeviceRef = Field(default_factory=DeviceRef.CPU)
    dtype: DType = DType.bfloat16


class AutoencoderKLQwenImageConfig(AutoencoderKLQwenImageConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLQwenImageConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLQwenImageConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return AutoencoderKLQwenImageConfig(**init_dict)


class AutoencoderKLFlux2Config(AutoencoderKLConfigBase):
    patch_size: tuple[int, int] = (2, 2)
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    latent_channels: int = 32  # Flux2 uses 32 channels, Flux1 uses 4

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "AutoencoderKLFlux2Config":
        """Generate AutoencoderKLFlux2Config from dictionary.

        Args:
            config_dict: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.

        Returns:
            AutoencoderKLFlux2Config instance.
        """
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in AutoencoderKLConfigBase.__annotations__
        }
        # Add Flux2-specific parameters if present
        flux2_params = ["patch_size", "batch_norm_eps", "batch_norm_momentum"]
        for param in flux2_params:
            if param in config_dict:
                init_dict[param] = config_dict[param]
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )

        if init_dict.get("shift_factor") is None:
            init_dict["shift_factor"] = 0.0
        if (
            "scaling_factor" in init_dict
            and float(init_dict["scaling_factor"]) == 0.0
        ):
            raise ValueError("`scaling_factor` must be non-zero.")

        return AutoencoderKLFlux2Config(**init_dict)
