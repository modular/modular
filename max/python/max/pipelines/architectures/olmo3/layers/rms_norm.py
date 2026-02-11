# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""RMSNorm implementation for Olmo3 models."""

from max.nn.norm import RMSNorm, rms_norm
from max.tensor import Tensor


class Olmo3RMSNorm(RMSNorm):
    """RMSNorm implementation for Olmo3 models.
    Similar to the traditional RMSNorm, but does (x * w).to(orig_dtype) instead
    of x.to(orig_dtype) * w.
    """

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(
            x,
            self.weight,
            self.eps,
            weight_offset=0.0,
            multiply_before_cast=True,
        )
