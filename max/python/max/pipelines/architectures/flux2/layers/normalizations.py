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

"""Normalization layers for Flux2."""

from max import functional as F
from max.nn import Linear, Module, module_dataclass
from max.nn.norm import rms_norm
from max.tensor import Tensor


@module_dataclass
class WeightedRMSNorm(Module):
    """RMSNorm wrapper for Flux2.

    Applies Root Mean Square Layer Normalization to the input tensor.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        """Initialize RMSNorm.

        Args:
            normalized_shape: shape of the input to normalize.
            eps: Small value for numerical stability.
            elementwise_affine: If True, learn affine parameters.
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        if elementwise_affine:
            self.weight = Tensor.ones(self.normalized_shape)
        else:
            self.weight = None
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def __call__(self, x: Tensor) -> Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape as input.
        """
        return rms_norm(
            x,
            self.weight,
            self.eps,
            weight_offset=0.0,
            multiply_before_cast=self.elementwise_affine,
        )


class WeightedLayerNorm(Module):
    """LayerNorm for Flux2.

    Standard Layer Normalization with optional affine transformation.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        """Initialize LayerNorm.

        Args:
            normalized_shape: Input shape to normalize (typically last dimension).
            eps: Small value for numerical stability.
            elementwise_affine: If True, learn affine parameters (weight and bias).
            bias: Whether to use bias in the linear projection.
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Tensor.ones(list(normalized_shape))
            if bias:
                self.bias = Tensor.zeros(list(normalized_shape))
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        """Apply LayerNorm.

        Args:
            x: Input tensor where the last len(normalized_shape) dimensions will be normalized.

        Returns:
            Normalized tensor of same shape as input.
        """
        gamma = (
            self.weight
            if self.weight is not None
            else Tensor.ones(
                self.normalized_shape, dtype=x.dtype, device=x.device
            )
        )
        bias = (
            self.bias
            if self.bias is not None
            else Tensor.zeros(
                self.normalized_shape, dtype=x.dtype, device=x.device
            )
        )
        return F.layer_norm(
            x,
            gamma=gamma,
            beta=bias,
            epsilon=self.eps,
        )


class AdaLayerNormContinuous(Module):
    """Adaptive Layer Normalization with continuous conditioning.

    Used for the final output normalization in Flux2, where the normalization
    is conditioned on the timestep embedding.
    """

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
        if norm_type == "layer_norm":
            self.norm = WeightedLayerNorm(
                embedding_dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )
        elif norm_type == "rms_norm":
            self.norm = WeightedRMSNorm(
                embedding_dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'rms_norm'."
            )

    def __call__(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
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
