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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm


class LayerNormNoAffine(Module):
    """LayerNorm over the last dimension without learned affine parameters."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: TensorValue) -> TensorValue:
        dim = x.shape[-1]
        gamma = ops.broadcast_to(
            ops.constant(1.0, x.dtype, device=x.device),
            shape=[dim],
        )
        beta = ops.broadcast_to(
            ops.constant(0.0, x.dtype, device=x.device),
            shape=[dim],
        )
        return ops.layer_norm(
            x,
            gamma=gamma,
            beta=beta,
            epsilon=self.eps,
        )


class AdaLayerNormContinuous(Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ) -> None:
        """Initialize adaptive layer normalization continuous module.

        Args:
            embedding_dim: Embedding dimension to use during projection.
            conditioning_embedding_dim: Dimension of the input condition.
            dtype: Weight dtype.
            device: Weight device.
            eps: Epsilon factor for normalization.
            bias: Whether to use bias in linear projection.
            norm_type: Type of normalization to use ("layer_norm" or "rms_norm").
        """
        super().__init__()
        self.linear = Linear(
            in_dim=conditioning_embedding_dim,
            out_dim=embedding_dim * 2,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        if norm_type == "layer_norm":
            self.norm: Module = LayerNormNoAffine(eps=eps)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, dtype=dtype, eps=eps)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def __call__(
        self,
        x: TensorValue,
        conditioning_embedding: TensorValue,
    ) -> TensorValue:
        """Apply adaptive layer normalization with conditioning.

        Args:
            x: Input tensor.
            conditioning_embedding: Conditioning embedding tensor.

        Returns:
            Normalized and conditioned tensor.
        """
        emb = self.linear(ops.cast(ops.silu(conditioning_embedding), x.dtype))
        width = x.shape[-1]
        scale, shift = ops.split(emb, [width, width], axis=1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
