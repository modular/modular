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
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.norm import LayerNorm, RMSNorm


class AdaLayerNormZeroSingle(Module[..., tuple[Tensor, Tensor]]):
    """Adaptive layer normalization zero single module."""

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
    ):
        """Initialize adaptive layer normalization zero single module.

        Args:
            embedding_dim: Size of each embedding vector.
            norm_type: Type of normalization to use ("layer_norm").
            bias: Whether to use bias in linear projection.

        """
        super().__init__()
        self.linear = Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=bias,
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(
                embedding_dim,
                use_bias=False,
                eps=1e-6,
                keep_dtype=True,
                elementwise_affine=False,
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self, x: Tensor, emb: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor.
            emb: Optional embedding tensor for conditioning.

        Returns:
            Tuple of normalized tensor and gate values.
        """
        emb = self.linear(F.silu(emb))
        shift_msa, scale_msa, gate_msa = F.chunk(emb, 3, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormZero(
    Module[..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
):
    r"""Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int | None = None,
        norm_type: str = "layer_norm",
        bias: bool = True,
    ):
        """Initialize adaptive layer normalization zero module.

        Args:
            embedding_dim: Size of each embedding vector.
            num_embeddings: Optional size of the embeddings dictionary.
            norm_type: Type of normalization to use ("layer_norm" or "fp32_layer_norm").
            bias: Whether to use bias in linear projection.
        """
        super().__init__()
        if num_embeddings is not None:
            # self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
            raise NotImplementedError(
                "CombinedTimestepLabelEmbeddings is not implemented"
            )
        else:
            self.emb = None

        self.linear = Linear(
            embedding_dim,
            6 * embedding_dim,
            bias=bias,
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(
                embedding_dim,
                use_bias=False,
                eps=1e-6,
                keep_dtype=True,
                elementwise_affine=False,
            )
        elif norm_type == "fp32_layer_norm":
            # self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
            raise NotImplementedError("FP32LayerNorm is not implemented")
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: Tensor,
        timestep: Tensor | None = None,
        class_labels: Tensor | None = None,
        hidden_dtype: DType | None = None,
        emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply adaptive layer normalization with gate values for attention and MLP.

        Args:
            x: Input tensor.
            timestep: Optional timestep tensor.
            class_labels: Optional class label tensor.
            hidden_dtype: Optional hidden data type.
            emb: Optional embedding tensor for conditioning.

        Returns:
            Tuple of (normalized tensor, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        """
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(F.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            F.chunk(emb, 6, axis=1)
        )
        x = self.norm(x)
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(Module[[Tensor, Tensor], Tensor]):
    r"""Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Boolean flag to denote if bias should be used.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    norm: LayerNorm | RMSNorm

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        """Initialize adaptive layer normalization continuous module.

        Args:
            embedding_dim: Embedding dimension to use during projection.
            conditioning_embedding_dim: Dimension of the input condition.
            eps: Epsilon factor for normalization.
            bias: Whether to use bias in linear projection.
            norm_type: Type of normalization to use ("layer_norm" or "rms_norm").
        """
        super().__init__()
        self.silu = F.silu
        self.linear = Linear(
            conditioning_embedding_dim,
            embedding_dim * 2,
            bias=bias,
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(
                embedding_dim,
                eps=eps,
                keep_dtype=True,
                elementwise_affine=False,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        """Apply adaptive layer normalization with conditioning.

        Args:
            x: Input tensor.
            conditioning_embedding: Conditioning embedding tensor.

        Returns:
            Normalized and conditioned tensor.
        """
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(F.cast(self.silu(conditioning_embedding), x.dtype))
        scale, shift = F.chunk(emb, 2, axis=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
