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
"""Inkling 2 normalizations, computed in float32."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn.norm import RMSNorm
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor


class Inkling2RMSNorm(RMSNorm):
    """RMSNorm that takes any float input and returns bfloat16.

    Hugging Face's model normalizes and scales in float32 and rounds once,
    so a float32 stream (the attn_res mixture) and a bfloat16 one give the
    same rounding point.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes over the last axis, scaling before the cast back."""
        normed = rms_norm(
            x.cast(DType.float32),
            self.weight.cast(DType.float32),
            self.eps,
            multiply_before_cast=True,
        )
        return normed.cast(DType.bfloat16)


def unweighted_rms_norm(x: Tensor, eps: float) -> Tensor:
    """Returns ``x * rsqrt(mean(x ** 2) + eps)`` over the last axis."""
    return x * F.rsqrt(F.mean(x * x, axis=-1) + eps)
