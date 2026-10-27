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
"""Laguna RMSNorm (ModuleV3)."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.norm.rms_norm import rms_norm
from max.experimental.tensor import Tensor


class LagunaRMSNorm(RMSNorm):
    """Llama-style RMSNorm whose gamma keeps the checkpoint's storage dtype.

    The ModuleV3 loader requires a parameter's declared dtype to match the
    checkpoint's, and Laguna checkpoints do not all store their norms in the
    activation dtype. The gamma is cast to the input dtype at use, as the graph
    architecture's ``RMSNorm`` does.

    Args:
        dim: Size of the normalized last dimension.
        eps: Epsilon added to the mean of squares.
        dtype: Storage dtype of the gamma.
    """

    def __init__(self, dim: int, eps: float, dtype: DType) -> None:
        super().__init__(dim, eps=eps)
        self.weight = Tensor.ones([dim], dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight.cast(x.dtype), self.eps)
