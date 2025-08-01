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

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import LayerNorm, Module

from ..model_config import Idefics3VisionConfig
from .connector import Idefics3Connector
from .embeddings import Idefics3VisionEmbeddings
from .encoder import Idefics3VisionEncoder


class Idefics3VisionModel(Module):
    """Vision transformer model for processing images in Idefics3.

    This implements the vision encoder component of Idefics3, which processes
    input images and produces visual embeddings that can be consumed by the
    language model. The architecture follows a standard Vision Transformer (ViT)
    design with Idefics3-specific enhancements:

    Key components:
    - Patch embedding layer that converts image patches to embeddings
    - Stack of transformer encoder layers

    Note:
        Currently limited to single-device execution.
    """

    def __init__(
        self, config: Idefics3VisionConfig, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.dtype = dtype
        self.device = device

        self.embeddings = Idefics3VisionEmbeddings(config, dtype, device)
        self.encoder = Idefics3VisionEncoder(config, dtype, device)
        self.patch_size = config.patch_size
        self.post_layernorm = LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.connector = Idefics3Connector(config, dtype, device)

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: Input pixel values tensor.

        Returns:
            Image embeddings tensor, flattened for language model.
        """

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=None
        )

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=None,
        )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.connector(hidden_states)

        # Reshape to flatten spatial dimensions, keeping the last dimension (feature dimension)
        image_hidden_states = ops.reshape(
            hidden_states, (-1, hidden_states.shape[-1])
        )

        return image_hidden_states
