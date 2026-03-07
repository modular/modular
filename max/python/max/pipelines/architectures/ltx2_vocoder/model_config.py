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


class LTX2VocoderConfigBase(MAXModelConfigBase):
    hidden_channels: int = 1024
    in_channels: int = 128
    leaky_relu_negative_slope: float = 0.1
    out_channels: int = 2
    output_sampling_rate: int = 24000
    resnet_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    resnet_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    upsample_factors: tuple[int, ...] = (6, 5, 2, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 15, 8, 4, 4)
    dtype: DType = DType.float32  # Vocoders often run in float32
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> MAXModelConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2VocoderConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2VocoderConfig(**init_dict)


# TODO: convert the above to the below's format
# class LTX2VocoderConfigBase(MAXModelConfigBase):
#     vocab_size: int = 32128
#     d_model: int = 512
#     d_kv: int = 64
#     d_ff: int = 2048
#     num_layers: int = 6
#     num_decoder_layers: int | None = None
#     num_heads: int = 8
#     relative_attention_num_buckets: int = 32
#     relative_attention_max_distance: int = 128
#     dropout_rate: float = 0.1
#     layer_norm_epsilon: float = 1e-6
#     initializer_factor: float = 1.0
#     feed_forward_proj: str = "relu"
#     dense_act_fn: str | None = Field(default=None, exclude=True)
#     is_gated_act: bool = Field(default=False, exclude=True)
#     is_decoder: bool = Field(default=False, exclude=True)
#     is_encoder_decoder: bool = True
#     use_cache: bool = True
#     pad_token_id: int = 0
#     eos_token_id: int = 1
#     classifier_dropout: float = 0.0
#     device: DeviceRef = Field(default_factory=DeviceRef.GPU)
#     dtype: DType = DType.bfloat16


# class LTX2VocoderConfig(LTX2VocoderConfigBase):
#     @staticmethod
#     def generate(
#         config_dict: dict[str, Any],
#         encoding: SupportedEncoding,
#         devices: list[Device],
#     ) -> LTX2VocoderConfigBase:
#         init_dict = {
#             key: value
#             for key, value in config_dict.items()
#             if key in LTX2VocoderConfigBase.__annotations__
#         }
#         init_dict.update(
#             {
#                 "dtype": supported_encoding_dtype(encoding),
#                 "device": DeviceRef.from_device(devices[0]),
#             }
#         )
#         return LTX2VocoderConfigBase(**init_dict)
