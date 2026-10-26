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
# DOC: develop/layers.mdx

from max.driver import CPU
from max.experimental.nn.norm import RMSNorm, rms_norm
from max.experimental.tensor import Tensor, default_device


class MultiplyBeforeCastRMSNorm(RMSNorm):
    """RMSNorm that multiplies by the weight before casting back."""

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(
            x,
            self.weight,
            self.eps,
            weight_offset=0.0,
            multiply_before_cast=True,
        )


with default_device(CPU()):
    norm = MultiplyBeforeCastRMSNorm(8, eps=1e-6)
    print([name for name, _ in norm.parameters])
