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
from max.pipelines.lib.config_enums import supported_encoding_dtype
from pydantic import Field


class Flux2ConfigBase(MAXModelConfigBase):
    patch_size: int = 1
    in_channels: int = 128
    out_channels: int | None = None
    num_layers: int = 8
    num_single_layers: int = 48
    attention_head_dim: int = 128
    num_attention_heads: int = 48
    joint_attention_dim: int = 15360
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 2000
    eps: float = 1e-6
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class Flux2Config(Flux2ConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Flux2ConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in Flux2ConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return Flux2ConfigBase(**init_dict)
