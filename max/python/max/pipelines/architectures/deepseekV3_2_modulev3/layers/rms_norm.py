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
"""RMSNorm variant matching DeepseekV3.2's final norm."""

from __future__ import annotations

from max.experimental.nn.norm import RMSNorm, rms_norm
from max.experimental.tensor import Tensor


class MultiplyBeforeCastRMSNorm(RMSNorm):
    """RMSNorm that applies the weight before casting back to the input dtype.

    ``max.nn.RMSNorm`` defaults ``multiply_before_cast`` to True and
    ``ops.rms_norm`` defaults it to False, so the graph-API model's final norm
    -- the only one that does not pass the flag explicitly -- multiplies before
    the cast while a plain ModuleV3 ``RMSNorm`` multiplies after. The two round
    differently in bfloat16, directly on the tensor that feeds ``lm_head``.
    """

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)
