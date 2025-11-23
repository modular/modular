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
"""Implements the QwenImageEdit2511 multimodal model architecture."""

from max.nn import (
    Module,
)

from .model_config import QwenImageEdit2511Config
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .nn.autoencoderkl_qwenimage import AutoencoderKLQwenImage
from .nn.transformer_qwenimage import QwenImageTransformer2DModel
from max.pipelines.architectures.qwen2_5vl.model import Qwen2_5VLModel


class QwenImageEdit2511(Module):
    """The overall interface to the QwenImageEdit2511 model."""

    def __init__(self, config: QwenImageEdit2511Config) -> None:
        self.config = config
        self.scheduler = self.build_scheduler()
        self.vae = self.build_vae()
        self.text_encoder = self.build_text_encoder()
        self.transformer = self.build_transformer()

    def build_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Build the scheduler component."""
        return FlowMatchEulerDiscreteScheduler(self.config.scheduler_config)

    def build_vae(self) -> AutoencoderKLQwenImage:
        """Build the VAE component."""
        return AutoencoderKLQwenImage(self.config.vae_config)

    def build_text_encoder(self) -> Qwen2_5VLModel:
        """Build the text encoder component."""
        return Qwen2_5VLModel(**self.config.text_encoder_config)

    def build_transformer(self) -> QwenImageTransformer2DModel:
        """Build the transformer component."""
        return QwenImageTransformer2DModel(self.config.transformer_config)

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "QwenImageEdit2511 is a container class. Use scheduler(), vae(), text_encoder(), or transformer() instead"
        )
