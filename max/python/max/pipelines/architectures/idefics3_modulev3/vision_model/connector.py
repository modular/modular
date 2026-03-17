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
"""Idefics3 connector (ModuleV3).

Bridges vision and text modalities via pixel shuffle and modality projection.
"""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor

from ..model_config import Idefics3VisionConfig


class Idefics3SimpleMLP(Module[[Tensor], Tensor]):
    """Simple linear projection from vision to text embedding space."""

    def __init__(self, config: Idefics3VisionConfig) -> None:
        super().__init__()
        input_size = config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config_hidden_size

        self.proj = Linear(in_dim=input_size, out_dim=output_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Idefics3Connector(Module[[Tensor], Tensor]):
    """Connector bridging vision and text modalities (ModuleV3).

    Performs pixel shuffle to reduce spatial resolution (increasing embedding
    dimension) followed by a linear modality projection.
    """

    def __init__(self, config: Idefics3VisionConfig) -> None:
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(config=config)

    def pixel_shuffle(self, x: Tensor, scale_factor: int) -> Tensor:
        """Reduce spatial resolution while increasing embedding dimension.

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim].
            scale_factor: Factor by which to reduce spatial dimensions.

        Returns:
            Tensor of shape [batch, seq_len/(sf^2), embed_dim*(sf^2)].
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]

        height = width = int(int(seq_len) ** 0.5)

        x = F.reshape(x, [batch_size, height, width, embed_dim])
        new_width = width // scale_factor
        new_embed_dim_1 = embed_dim * scale_factor
        x = F.reshape(x, [batch_size, height, new_width, new_embed_dim_1])
        x = x.transpose(1, 2)
        new_height = height // scale_factor
        final_embed_dim = embed_dim * (scale_factor * scale_factor)
        x = F.reshape(x, [batch_size, new_width, new_height, final_embed_dim])
        x = x.transpose(1, 2)
        final_seq_len = seq_len // (scale_factor * scale_factor)
        x = F.reshape(x, [batch_size, final_seq_len, final_embed_dim])

        return x

    def forward(self, image_hidden_states: Tensor) -> Tensor:
        image_hidden_states = self.pixel_shuffle(
            image_hidden_states, self.scale_factor
        )
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states
