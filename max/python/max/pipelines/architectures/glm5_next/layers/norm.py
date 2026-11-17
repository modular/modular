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
"""The final norm of GLM-5.3-Flash's decoder stack, on ModuleV3."""

from __future__ import annotations

from max.nn.norm import RMSNorm, rms_norm
from max.tensor import Tensor

__all__ = ["MultiplyBeforeCastRMSNorm"]


class MultiplyBeforeCastRMSNorm(RMSNorm):
    """RMSNorm that applies the weight before casting back to the input dtype.

    A plain ModuleV3 ``RMSNorm`` multiplies after the cast. The final norm
    multiplies before it, as the graph-API build's did, and the two round
    differently in bfloat16, directly on the tensor that feeds ``lm_head``.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes ``x`` with the weight applied before the cast."""
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)
