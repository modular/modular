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
"""Implements the LTX2 video generation model architecture."""

from __future__ import annotations

from dataclasses import asdict

from max.driver import Device
from max.nn.module_v3 import Module
from max.pipelines.architectures.gemma3.gemma3 import Gemma3

from ..z_image_module_v3.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from .model_config import LTX2Config
from .nn.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from .nn.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
from .nn.transformer_ltx2 import LTX2Transformer3DModel


class LTX2(Module):
    """The overall interface to the LTX2 model."""

    def __init__(
        self, config: LTX2Config, device: Device | None = None
    ) -> None:
        self.config = config
        self.device = device
        self.scheduler = self.build_scheduler()
        self.vae = self.build_vae()
        self.vae_audio = self.build_vae_audio()
        # self.text_encoder = self.build_text_encoder()
        self.transformer = self.build_transformer()

    def build_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Build the scheduler component."""
        return FlowMatchEulerDiscreteScheduler(
            **asdict(self.config.scheduler_config)
        )

    def build_vae(self) -> AutoencoderKLLTX2Video:
        """Build the VAE component."""
        return AutoencoderKLLTX2Video(**asdict(self.config.vae_config))

    def build_vae_audio(self) -> AutoencoderKLLTX2Audio:
        """Build the VAE audio component."""
        return AutoencoderKLLTX2Audio(**asdict(self.config.vae_audio_config))

    def build_text_encoder(self) -> Gemma3:
        """Build the text encoder component.

        Uses native Gemma3 with return_hidden_states=ALL configured.
        The hidden states are extracted in model.py's _encode_prompt(),
        specifically hidden_states[-2] (second-to-last layer output).
        """
        return Gemma3(self.config.text_encoder_config)

    def build_transformer(self) -> LTX2Transformer3DModel:
        """Build the transformer component."""
        # Pass device for RoPE precomputation on GPU
        transformer_kwargs = asdict(self.config.transformer_config)
        transformer_kwargs["device"] = self.device
        return LTX2Transformer3DModel(**transformer_kwargs)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "LTX2 is a container class. Use scheduler(), vae(), text_encoder(), or transformer() instead"
        )
