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

"""Scaled Word Embedding for the ModuleV3 API."""

from max.experimental import functional as F
from max.experimental.nn.embedding import Embedding
from max.experimental.tensor import Tensor


class ScaledEmbedding(Embedding):
    """An Embedding that multiplies lookup results by a scale factor.

    Used by Gemma3 to scale token embeddings by ``sqrt(hidden_size)``.
    """

    embed_scale: float

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        embed_scale: float = 1.0,
    ) -> None:
        super().__init__(vocab_size, dim=dim)
        self.embed_scale = embed_scale

    def forward(self, indices: Tensor) -> Tensor:
        result = super().forward(indices)
        return result * F.constant(
            self.embed_scale, result.dtype, device=result.device
        )
