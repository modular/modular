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

"""Kimi K2.5 bespoke context object."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.core.context import TextAndVisionContext


@dataclass(kw_only=True)
class KimiK2_5TextAndVisionContext(TextAndVisionContext):
    """Context for Kimi K2.5 multimodal model processing.

    Extends ``TextAndVisionContext`` with the per-image grid dimensions
    (temporal, height, width) required by the vision encoder's positional
    embedding and RoPE computation.
    """

    grid_thws: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )
    """Per-image grid dimensions of shape ``(N, 3)`` where each row is
    ``(temporal_frames, height_patches, width_patches)``."""
