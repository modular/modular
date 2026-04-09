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
"""Gemma3 RMSNorm for the ModuleV3 API."""

from max.experimental.nn.norm import GemmaRMSNorm


class Gemma3RMSNorm(GemmaRMSNorm):
    """Gemma3 RMSNorm uses (1 + weight) as the scale factor with multiply_before_cast."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim=dim, eps=eps)
