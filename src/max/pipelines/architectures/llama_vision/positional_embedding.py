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

"""Llama 3.2 Transformer Vision Model Positional Embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue, Weight, ops
from max.nn import Embedding
from max.nn.layer import Layer


@dataclass
class PrecomputedAspectRatioEmbedding(Layer):
    """
    Llama 3.2 precomputed aspect ratio embedding.

    Args:
        params: Hyperparameters for this PrecomputedAspectRatioEmbedding layer.
        gate: The gating parameter to control the contribution of the aspect
              ratio embedding.
        embedding: The aspect ratio embedding.
    """

    max_num_tiles: int
    hidden_size: int
    gate: Weight
    embedding: Embedding
    is_gated: bool = False

    def __call__(
        self, hidden_state: TensorValue, aspect_ratio_ids: TensorValue
    ) -> TensorValue:
        batch_size, num_tiles, _, hidden_size = hidden_state.shape
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape((-1, num_tiles, 1, self.hidden_size))

        if self.is_gated:
            embeddings = embeddings * ops.tanh(self.gate)

        # We're broadcasting in the add operation below, so we call reshape()
        # on embeddings first.
        embeddings = embeddings.reshape(
            (
                batch_size,
                num_tiles,
                embeddings.shape[2],
                hidden_size,
            )
        )
        return hidden_state + embeddings


@dataclass
class PrecomputedPositionEmbedding(Layer):
    """
    Llama 3.2 precomputed position embedding.

    Args:
        gate: The gating parameter to control the contribution of the position
              embedding or another component of the model.
        embedding: The precomputed position embedding.
        tile_embedding: The embedding associated with tiles or patches in a
                        vision model.
    """

    image_size: int
    patch_size: int
    hidden_size: int
    max_num_tiles: int
    gate: Weight
    embedding: Weight
    tile_embedding: Embedding

    def __call__(
        self, hidden_state: TensorValue, aspect_ratio_ids: TensorValue
    ) -> TensorValue:
        # position embeddings
        gated_position_embedding = (1 - ops.tanh(self.gate)) * self.embedding

        num_patches = (self.image_size // self.patch_size) ** 2 + 1
        gated_position_embedding = gated_position_embedding.reshape(
            (
                1,
                1,
                num_patches,
                self.hidden_size,
            )
        )
        # We're broadcasting gated_position_embedding in the add operation below,
        # so we call reshape() on hidden_state first.
        batch_size, num_tiles, _, _ = hidden_state.shape
        hidden_state = hidden_state.reshape(
            (
                batch_size,
                num_tiles,
                num_patches,
                self.hidden_size,
            )
        )
        hidden_state = hidden_state + gated_position_embedding

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        tile_position_embedding = tile_position_embedding.reshape(
            (
                batch_size,
                self.max_num_tiles,
                num_patches,
                self.hidden_size,
            )
        )
        gated_tile_position_embedding = (
            ops.tanh(self.gate) * tile_position_embedding
        )
        # This explicit reshape is called only to match num_tiles dim in
        # tile_position_embedding.
        hidden_state = hidden_state.reshape(tile_position_embedding.shape)
        return hidden_state + gated_tile_position_embedding
