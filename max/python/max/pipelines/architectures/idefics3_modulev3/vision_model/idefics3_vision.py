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
"""Idefics3 vision model (ModuleV3)."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.norm import LayerNorm
from max.experimental.tensor import Tensor

from ..model_config import Idefics3VisionConfig
from .connector import Idefics3Connector
from .embeddings import Idefics3VisionEmbeddings
from .encoder import Idefics3VisionEncoder


class Idefics3VisionModel(Module[[Tensor], Tensor]):
    """Vision transformer for processing images in Idefics3 (ModuleV3).

    Processes input images through patch embeddings, a transformer encoder,
    post-layer normalization, and a connector that bridges vision and text
    embedding spaces.
    """

    def __init__(self, config: Idefics3VisionConfig) -> None:
        super().__init__()
        self.embeddings = Idefics3VisionEmbeddings(config)
        self.encoder = Idefics3VisionEncoder(config)
        self.post_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            use_bias=True,
        )
        self.connector = Idefics3Connector(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: Input pixel values [batch, channels, height, width].

        Returns:
            Image embeddings flattened for the language model
            [total_image_tokens, hidden_size].
        """
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.connector(hidden_states)

        # Flatten batch and spatial dims: [batch, seq, dim] -> [total, dim]
        image_hidden_states = F.reshape(
            hidden_states, (-1, hidden_states.shape[-1])
        )

        return image_hidden_states
