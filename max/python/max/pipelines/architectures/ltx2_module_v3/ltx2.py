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

from max.nn import (
    Module,
)

from .model_config import LTX2Config
from .nn.transformer_ltx2 import LTX2Transformer3DModel
from .nn.autoencoder_kl_ltx2 import AutoEncoderKL3DModel
from .nn.audio_transformer import AudioTransformer


class LTX2(Module):
    """The overall interface to the LTX2 model."""

    def __init__(self, config: LTX2Config) -> None:
        self.config = config
        self.transformer_3d = self.build_transformer_3d()
        self.vae = self.build_vae()
        self.audio_transformer = self.build_audio_transformer()

    def build_transformer(self) -> LTX2Transformer3DModel:
        return LTX2Transformer3DModel(
            config=self.config.transformer_3d_config,
        )

    def build_video_vae(self) -> AutoEncoderKL3DModel:
        return AutoEncoderKL3DModel(
            config=self.config.autoencoder_kl_3d_config,
        )

    def build_audio_vae(self) -> AudioTransformer:
        return AudioTransformer(
            config=self.config.audio_transformer_config,
        )

    def build_text_encoder(self) -> Gemma3ForConditionalGeneration:
        return Gemma3ForConditionalGeneration(config=self.config.text_encoder_config)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "LTX2 is a container class. Use transformer_3d() or vae() or audio_transformer() instead"
        )
