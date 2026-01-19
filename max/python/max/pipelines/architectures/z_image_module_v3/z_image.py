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

from dataclasses import asdict

from max.driver import Device
from max.nn.module_v3 import Module
from max.pipelines.architectures.qwen3.qwen3 import Qwen3

from .model_config import ZImageConfig
from .nn.autoencoder_kl import AutoencoderKL
from .nn.transformer_z_image import ZImageTransformer2DModel
from .scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class ZImage(Module):
    """The overall interface to the ZImage model."""

    def __init__(
        self, config: ZImageConfig, device: Device | None = None
    ) -> None:
        self.config = config
        self.device = device
        self.scheduler = self.build_scheduler()
        self.vae = self.build_vae()
        # self.text_encoder = self.build_text_encoder()
        self.transformer = self.build_transformer()

    def build_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Build the scheduler component."""
        return FlowMatchEulerDiscreteScheduler(
            **asdict(self.config.scheduler_config)
        )

    def build_vae(self) -> AutoencoderKL:
        """Build the VAE component."""
        return AutoencoderKL(**asdict(self.config.vae_config))

    def build_text_encoder(self) -> Qwen3:
        """Build the text encoder component.

        Uses native Qwen3 with return_hidden_states=ALL configured.
        The hidden states are extracted in model.py's _encode_prompt(),
        specifically hidden_states[-2] (second-to-last layer output).
        """
        return Qwen3(self.config.text_encoder_config)

    def build_transformer(self) -> ZImageTransformer2DModel:
        """Build the transformer component."""
        # Pass device for RoPE precomputation on GPU
        transformer_kwargs = asdict(self.config.transformer_config)
        transformer_kwargs["device"] = self.device
        return ZImageTransformer2DModel(**transformer_kwargs)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "ZImage is a container class. Use scheduler(), vae(), text_encoder(), or transformer() instead"
        )
