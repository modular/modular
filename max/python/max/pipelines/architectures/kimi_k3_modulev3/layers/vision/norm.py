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
"""PyTorch-compatible RMS normalization for the Kimi K3 vision tower."""

from __future__ import annotations

from max.nn.norm.rms_norm import RMSNorm, rms_norm
from max.tensor import Tensor


class VisionRMSNorm(RMSNorm):
    """Multiplies the scale before casting the normalized values back."""

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes with the checkpoint's PyTorch multiplication order."""
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)
