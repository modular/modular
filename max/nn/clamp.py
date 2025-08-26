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

from __future__ import annotations

from max.graph import TensorValue, ops


def clamp(
    x: TensorValue, min: float | None = None, max: float | None = None
) -> TensorValue:
    """Clamps values in `x` to `[min, max]`

    Args:
        x: Input tensor to clamp.
        min: Minimum value. If None, no lower bound is applied.
        max: Maximum value. If None, no upper bound is applied.

    Returns:
        Clamped tensor.
    """
    if min is not None:
        x = ops.max(x, ops.constant(min, x.dtype, device=x.type.device))
    if max is not None:
        x = ops.min(x, ops.constant(max, x.dtype, device=x.type.device))
    return x
