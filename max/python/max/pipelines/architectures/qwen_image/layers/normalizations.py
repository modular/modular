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

"""Normalization layers for QwenImage transformer (module v2)."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm


class LayerNormNoAffine(Module):
    """LayerNorm over the last dimension without learned affine parameters."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: TensorValue) -> TensorValue:
        """Normalizes ``x`` without a learned scale or shift."""
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
    """Layer norm whose scale and shift come from the conditioning vector."""

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
        """Normalizes ``x``, scaling and shifting it by the conditioning."""
        emb = self.linear(ops.cast(ops.silu(conditioning_embedding), x.dtype))
        width = x.shape[-1]
        scale, shift = ops.split(emb, [width, width], axis=1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
