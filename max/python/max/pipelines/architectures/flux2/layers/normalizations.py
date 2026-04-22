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
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm


class LayerNorm(Module):
    """Compatibility shim for the experimental LayerNorm API."""

    weight: Weight | None
    bias: Weight | None

    def __init__(
        self,
        dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
        keep_dtype: bool = True,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.keep_dtype = keep_dtype
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias
        if elementwise_affine:
            self.weight = Weight("weight", dtype, (dim,), device=device)
            self.bias = (
                Weight("bias", dtype, (dim,), device=device)
                if use_bias
                else None
            )
        else:
            self.weight = None
            self.bias = None

    def _affine_params(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        if self.weight is None:
            gamma = ops.broadcast_to(
                ops.constant(1.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            gamma = self.weight

        if self.bias is None:
            beta = ops.broadcast_to(
                ops.constant(0.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            beta = self.bias

        return gamma, beta

    def __call__(self, x: TensorValue) -> TensorValue:
        gamma, beta = self._affine_params(x)
        if self.keep_dtype:
            return ops.layer_norm(
                x,
                gamma=gamma,
                beta=beta,
                epsilon=self.eps,
            )

        output = ops.layer_norm(
            ops.cast(x, DType.float32),
            gamma=ops.cast(gamma, DType.float32),
            beta=ops.cast(beta, DType.float32),
            epsilon=self.eps,
        )
        return ops.cast(output, x.dtype)


class AdaLayerNormContinuous(Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        *,
        dtype: DType,
        device: DeviceRef,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ) -> None:
        """Initialize AdaLayerNormContinuous.

        Args:
            embedding_dim: Dimension of the input embeddings to normalize.
            conditioning_embedding_dim: Dimension of the conditioning embeddings.
            elementwise_affine: If True, learn affine parameters.
            dtype: Weight dtype.
            device: Weight device.
            eps: Small value for numerical stability in LayerNorm.
            bias: Whether to use bias in the linear projection.
            norm_type: Type of normalization to use ("layer_norm" or "rms_norm").
        """
        super().__init__()
        self.silu = ops.silu
        self.linear = Linear(
            in_dim=conditioning_embedding_dim,
            out_dim=embedding_dim * 2,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        self.norm: LayerNorm | RMSNorm
        if norm_type == "layer_norm":
            self.norm = LayerNorm(
                embedding_dim,
                dtype=dtype,
                device=device,
                eps=eps,
                elementwise_affine=elementwise_affine,
                use_bias=bias,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, dtype=dtype, eps=eps)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'rms_norm'."
            )

    def __call__(
        self,
        x: TensorValue,
        conditioning_embedding: TensorValue,
    ) -> TensorValue:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape [B, S, D].
            conditioning_embedding: Conditioning embedding (timestep) of shape [B, D_cond].

        Returns:
            Normalized and modulated tensor of shape [B, S, D].
        """
        conditioning_embedding = ops.cast(conditioning_embedding, x.dtype)
        emb = self.linear(self.silu(conditioning_embedding))

        scale, shift = ops.chunk(emb, chunks=2, axis=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
