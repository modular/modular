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
"""Implements the Qwen2.5VL multimodal model architecture."""

from __future__ import annotations

from max.nn.layer import Module
from max.pipelines.modeling.config_enums import supported_encoding_dtype

from .model_config import Qwen2_5VLConfig
from .nn.decoder import Qwen25VLDecoder
from .nn.visual_transformer import VisionTransformer


class Qwen2_5VL(Module):
    """The overall interface to the Qwen2.5VL model."""

    def __init__(self, config: Qwen2_5VLConfig) -> None:
        self.config = config
        self.vision_encoder = self.build_vision_encoder()
        self.language_model = self.build_language_model()

    def build_vision_encoder(self) -> VisionTransformer:
        return VisionTransformer(
            config=self.config.vision_config,
        )

    def build_language_model(self) -> Qwen25VLDecoder:
        """Return the language model component."""
        encoding = self.config.quantization_encoding
        assert encoding is not None, "quantization_encoding must be resolved"
        return Qwen25VLDecoder(
            self.config.llm_config,
            mrope_section=self.config.mrope_section,
            # The pipeline model's dtype, which the vision graph's outputs
            # are cast to before they reach the language graph.
            image_embedding_dtype=supported_encoding_dtype(encoding),
        )

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "Qwen2_5VL is a container class. Use vision_encoder() or language_model() instead"
        )
