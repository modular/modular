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
from max.nn import Module

from ..deepseekV3.deepseekV3 import DeepseekV3
from .layers.language_model import KimiDecoder
from .layers.vision.transformer import Transformer
from .model_config import (
    KimiK2_5Config,
)


class KimiK2_5(Module):
    """The overall interface to the KimiK2_5 model."""

    vision_encoder: Module

    def __init__(self, config: KimiK2_5Config) -> None:
        self.config = config
        self.vision_encoder = self.build_vision_encoder()
        self.language_model = self.build_language_model()

    def build_vision_encoder(self) -> Transformer:
        return Transformer(self.config.vision_config)

    def build_language_model(self) -> DeepseekV3:
        """Return the language model component."""
        return KimiDecoder(self.config.llm_config)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "KimiK2_5 is a container class. Use vision_encoder() or language_model() instead"
        )
