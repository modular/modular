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

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.norm import LayerNorm, RMSNorm


class AdaLayerNormContinuous(Module[[Tensor, Tensor], Tensor]):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        """Initialize AdaLayerNormContinuous.

        Args:
            embedding_dim: Dimension of the input embeddings to normalize.
            conditioning_embedding_dim: Dimension of the conditioning embeddings.
            elementwise_affine: If True, learn affine parameters.
            eps: Small value for numerical stability in LayerNorm.
            bias: Whether to use bias in the linear projection.
            norm_type: Type of normalization to use ("layer_norm" or "rms_norm").
        """
        self.silu = F.silu
        self.linear = Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        self.norm: LayerNorm | RMSNorm
        if norm_type == "layer_norm":
            self.norm = LayerNorm(
                embedding_dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
                use_bias=bias,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'rms_norm'."
            )

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape [B, S, D].
            conditioning_embedding: Conditioning embedding (timestep) of shape [B, D_cond].

        Returns:
            Normalized and modulated tensor of shape [B, S, D].
        """
        emb = self.linear(self.silu(conditioning_embedding))

        scale, shift = F.chunk(emb, 2, axis=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
