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

from typing import TYPE_CHECKING, Any

from max.driver import Device, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


class FluxConfigBase(MAXModelConfigBase):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class FluxConfig(FluxConfigBase):
    @classmethod
    def initialize(cls, pipeline_config: "PipelineConfig") -> Self:
        """Initialize FluxConfig from a PipelineConfig.

        Args:
            pipeline_config: The pipeline configuration.

        Returns:
            An initialized FluxConfig instance.
        """
        if pipeline_config.model.quantization_encoding is None:
            raise ValueError("Quantization encoding is required for FluxConfig")

        # Get the huggingface config if available
        hf_config = pipeline_config.model.huggingface_config
        config_dict = hf_config.to_dict() if hf_config is not None else {}

        # Convert device specs to devices
        devices = load_devices(pipeline_config.model.device_specs)

        # Generate config using the existing generate method
        config_base = cls.generate(
            config_dict,
            pipeline_config.model.quantization_encoding,
            devices,
        )

        # Convert to FluxConfig (which is just FluxConfigBase with extra methods)
        return cls(**config_base.model_dump())

    def get_max_seq_len(self) -> int:
        """Get the maximum sequence length.

        For pixel generation models, this returns a placeholder value
        as sequence length is not applicable.

        Returns:
            A placeholder sequence length value.
        """
        # Pixel generation models don't have a text sequence length constraint
        # Return a reasonable default
        return (
            77  # Standard CLIP text encoder max length used in diffusion models
        )

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> FluxConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in FluxConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return FluxConfigBase(**init_dict)
