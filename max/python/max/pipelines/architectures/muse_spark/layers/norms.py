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

"""Muse Spark RMSNorm variants for the ModuleV3 API."""

from __future__ import annotations

from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping


def weightless_rms_norm(x: Tensor, eps: float) -> Tensor:
    """Normalizes ``x`` over its last axis with no learned scale."""
    # Built on x's mesh, so the norm runs on whichever devices x lives on.
    ones = Tensor.ones(
        [x.shape[-1]], dtype=x.dtype, device=DeviceMapping.replicated(x.mesh)
    )
    return rms_norm(x, ones, eps, multiply_before_cast=True)


def input_scaled_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Scales ``x`` by ``1 + weight``, rounds to ``x``'s dtype, then normalizes."""
    return weightless_rms_norm(x * (1.0 + weight), eps)


class WeightlessRMSNorm(Module[[Tensor], Tensor]):
    """RMSNorm without a scale (qk, o, embedding and post-attention norms)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return weightless_rms_norm(x, self.eps)


class CenteredRMSNorm(Module[[Tensor], Tensor]):
    """Normalizes, then scales by ``1 + weight``."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = Tensor.zeros([dim])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # The reference rounds 1 + weight to the weight's dtype before the
        # fp32 multiply; weight_offset=1 would add it in fp32 instead.
        return rms_norm(
            x, 1.0 + self.weight, self.eps, multiply_before_cast=True
        )


class InputScaledRMSNorm(Module[[Tensor], Tensor]):
    """Scales by ``1 + weight``, rounds to ``x``'s dtype, then normalizes."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = Tensor.zeros([dim])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return input_scaled_rms_norm(x, self.weight, self.eps)
