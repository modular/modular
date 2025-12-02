# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements the ZImage multimodal model architecture."""

from __future__ import annotations

from max.nn import Module

from .model_config import ZImageConfig
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .nn.autoencoder_kl import AutoencoderKL
from .nn.transformer_z_image import ZImageTransformer2DModel
from max.pipelines.architectures.qwen3.model import Qwen3Model


class ZImage(Module):
    """The overall interface to the ZImage model."""

    def __init__(self, config: ZImageConfig) -> None:
        self.config = config
        self.scheduler = self.build_scheduler()
        self.vae = self.build_vae()
        self.text_encoder = self.build_text_encoder()
        self.transformer = self.build_transformer()

    def build_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Build the scheduler component."""
        return FlowMatchEulerDiscreteScheduler(self.config.scheduler_config)

    def build_vae(self) -> AutoencoderKL:
        """Build the VAE component."""
        return AutoencoderKL(self.config.vae_config)

    def build_text_encoder(self) -> Qwen3Model:
        """Build the text encoder component."""
        return Qwen3Model(**self.config.text_encoder_config)

    def build_transformer(self) -> ZImageTransformer2DModel:
        """Build the transformer component."""
        return ZImageTransformer2DModel(self.config.transformer_config)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "ZImage is a container class. Use scheduler(), vae(), text_encoder(), or transformer() instead"
        )
