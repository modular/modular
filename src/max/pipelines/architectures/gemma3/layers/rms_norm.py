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

from max.nn.norm.rms_norm import RMSNormV2


class Gemma3RMSNorm(RMSNormV2):
    def __init__(self, dim: int, eps: float = 1e-6):
        # Gemma3 uses (1.0 + weight) as the scale factor
        super().__init__(dim=dim, eps=eps, weight_offset=1.0)
