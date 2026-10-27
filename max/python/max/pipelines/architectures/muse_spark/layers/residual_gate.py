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

"""Muse Spark residual gate for the ModuleV3 API."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.tensor import Tensor


class MuseSparkResidualGate(Module[[Tensor, Tensor], Tensor]):
    """Mixes the residual and the branch output with a learned per-channel gate.

    ``out = alpha * residual + beta * h`` with ``beta = sigmoid(gate / T)``
    and ``alpha = sqrt(max(sigmoid(-gate / T) * (1 + beta), eps))``.
    """

    def __init__(
        self, hidden_size: int, temperature: float, eps: float = 1e-3
    ) -> None:
        super().__init__()
        self.gate = Tensor.zeros([hidden_size])
        self.temperature = temperature
        self.eps = eps

    def forward(self, residual: Tensor, h: Tensor) -> Tensor:
        # The reference keeps the gate and the mix in fp32.
        w = self.gate.cast(DType.float32) / self.temperature
        beta = F.sigmoid(w)
        alpha = F.sqrt(F.max(F.sigmoid(-w) * (1.0 + beta), self.eps))
        out = alpha * residual.cast(DType.float32) + beta * h.cast(
            DType.float32
        )
        return out.cast(residual.dtype)
