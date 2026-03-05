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
from .layers.vision.encoder import Encoder
from .model_config import (
    KimiK2_5Config,
    VisionConfig,
)


class KimiK2_5(Module):
    """The overall interface to the KimiK2_5 model."""

    vision_encoder: Module
    multimodal_projector: Module

    def __init__(self, config: KimiK2_5Config) -> None:
        self.config = config
        self.vision_encoder = self.build_vision_encoder()
        self.language_model = self.build_language_model()

    def build_vision_encoder(self) -> Module:
        vc: VisionConfig = self.config.vision_config
        return Encoder(
            num_heads=vc.vt_num_attention_heads,
            hidden_dim=vc.vt_hidden_size,
            mlp_dim=vc.vt_intermediate_size,
            num_layers=vc.vt_num_hidden_layers,
            rope_max_height=512,
            rope_max_width=512,
            rope_theta=10000.0,
            dtype=vc.dtype,
            device=vc.devices[0],
            has_bias=True,  # checkpoint contains biases
        )

    def build_language_model(self) -> DeepseekV3:
        """Return the language model component."""
        return DeepseekV3(self.config.llm_config)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "KimiK2_5 is a container class. Use vision_encoder() or language_model() instead"
        )
