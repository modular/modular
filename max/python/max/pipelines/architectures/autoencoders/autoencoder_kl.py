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
from max.graph import TensorValue
from max.graph.weights import Weights
from max.nn.layer import Module
from max.pipelines.lib import SupportedEncoding

from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLConfig
from .vae import Decoder


class AutoencoderKL(Module):
    r"""A VAE model with KL loss for decoding latent representations into images."""

    def __init__(self, config: AutoencoderKLConfig) -> None:
        super().__init__()
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            norm_type="group",
            mid_block_add_attention=config.mid_block_add_attention,
            use_post_quant_conv=config.use_post_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )

    def __call__(
        self, z: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        return self.decoder(z, temb)


class AutoencoderKLModel(BaseAutoencoderModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLConfig,
            autoencoder_class=AutoencoderKL,
            **kwargs,
        )
