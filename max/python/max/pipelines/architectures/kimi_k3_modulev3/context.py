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

"""Carries Kimi K3 image grids beside the decoder's token state."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.context import TextAndVisionContext


@dataclass(kw_only=True, slots=True)
class KimiK3TextAndVisionContext(TextAndVisionContext):
    """Stores per-image grids and patch rotary positions in media order."""

    grid_thws: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )
    position_ids: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    image_token_indices: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
