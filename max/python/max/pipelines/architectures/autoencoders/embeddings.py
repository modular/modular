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

# TODO: This file should be moved to a more general location.

import max.experimental.functional as F
from max.dtype import DType
from max.experimental import nn
from max.experimental.tensor import Tensor

from ..flux2.layers.embeddings import TimestepEmbedding, Timesteps


class PixArtAlphaCombinedTimestepSizeEmbeddings(
    nn.Module[[Tensor, Tensor, Tensor, int, DType], Tensor]
):
    """Combined embeddings for PixArt-Alpha (and LTX2 autoencoders)."""

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
        use_additional_conditions: bool = False,
    ) -> None:
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(
                num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
            )
            self.resolution_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=size_emb_dim
            )
            self.aspect_ratio_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=size_emb_dim
            )

    def forward(
        self,
        timestep: Tensor,
        resolution: Tensor,
        aspect_ratio: Tensor,
        batch_size: int,
        hidden_dtype: DType,
    ) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            F.cast(timesteps_proj, hidden_dtype)
        )

        if self.use_additional_conditions:
            res_emb = self.additional_condition_proj(F.flatten(resolution))
            res_emb = self.resolution_embedder(F.cast(res_emb, hidden_dtype))
            res_emb = F.reshape(res_emb, (batch_size, -1))

            ar_emb = self.additional_condition_proj(F.flatten(aspect_ratio))
            ar_emb = self.aspect_ratio_embedder(F.cast(ar_emb, hidden_dtype))
            ar_emb = F.reshape(ar_emb, (batch_size, -1))

            conditioning = timesteps_emb + F.concat([res_emb, ar_emb], axis=1)
        else:
            conditioning = timesteps_emb

        return conditioning
