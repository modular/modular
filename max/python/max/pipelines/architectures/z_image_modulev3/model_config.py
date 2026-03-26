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
from typing_extensions import Self


class ZImageConfig(MAXModelConfigBase):
    all_patch_size: tuple[int, ...] = (2,)
    all_f_patch_size: tuple[int, ...] = (1,)
    in_channels: int = 16
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (1024, 512, 512)
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)

    @classmethod
    def initialize_from_config(
        cls,
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Self:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in cls.model_fields
        }
        # Ignore omni-only fields in phase 1 (may appear in full checkpoints).
        init_dict.pop("siglip_feat_dim", None)

        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return cls(**init_dict)

    def fbcache_dims(self) -> tuple[int, int]:
        """(hidden_dim, output_dim) per image token for FBCache / Taylor tensors."""
        out_dim = (
            self.all_patch_size[0]
            * self.all_patch_size[0]
            * self.all_f_patch_size[0]
            * self.in_channels
        )
        return self.dim, out_dim


# Back-compat alias for call sites that refer to the config as a "base" type.
ZImageConfigBase = ZImageConfig
