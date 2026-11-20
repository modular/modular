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
"""The two RMS normalizations DeepSeek-V4 uses."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

__all__ = ["DeepseekV4RMSNorm", "weightless_rms_normalize"]


class DeepseekV4RMSNorm(Module[[Tensor], Tensor]):
    """RMSNorm with a float32 weight, computed in the input's dtype.

    The V2 ``max.nn.legacy.RMSNorm`` semantics, which differ from the ModuleV3
    ``RMSNorm``: the weight is cast to the input dtype first, and the scale is
    applied before the cast back (``multiply_before_cast``).
    """

    weight: Tensor

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = Tensor.ones([dim], dtype=DType.float32)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """RMS-normalizes ``x`` over the last axis, scaled by the weight."""
        return rms_norm(
            x, self.weight.cast(x.dtype), self.eps, multiply_before_cast=True
        )


def weightless_rms_normalize(x: Tensor, eps: float) -> Tensor:
    """``x * rsqrt(mean(x^2) + eps)`` over the last axis, with no gain."""
    x32 = x.cast(DType.float32)
    scale = F.rsqrt((x32 * x32).mean(axis=-1) + eps)
    return (x32 * scale).cast(x.dtype)
