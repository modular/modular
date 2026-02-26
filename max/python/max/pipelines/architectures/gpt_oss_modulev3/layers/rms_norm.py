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
"""RMSNorm implementation for GPT-OSS models."""

from max.experimental.tensor import Tensor
from max.nn.module_v3.norm import RMSNorm, rms_norm


class GptOssRMSNorm(RMSNorm):
    """RMSNorm implementation for GPT-OSS models.
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
