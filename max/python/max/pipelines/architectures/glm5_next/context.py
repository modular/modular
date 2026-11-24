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
"""GLM-5.3-Flash request context."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.context.context import TextAndVisionContext


@dataclass(kw_only=True, slots=True)
class Glm5NextTextAndVisionContext(TextAndVisionContext):
    """Adds each image's patch grid, which its pixels alone do not determine.

    Every request uses this context, text-only ones included: the pipeline's
    vision seam asserts a vision context for every request of a
    vision-capable architecture.
    """

    grid_thws: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )
    """``(num_images, 3)`` ``(t, h, w)`` patch grids, one row per entry in
    ``images``. The tower's rotary positions and attention segments are
    computed from these."""
