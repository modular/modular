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

"""Gemma4-specific context for storing prompt state."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.core import TextAndVisionContext


@dataclass(kw_only=True)
class Gemma4Context(TextAndVisionContext):
    """A context for storing prompt state for the Diancie model."""

    mm_token_type_ids: npt.NDArray[np.int64]
    pixel_position_ids: list[npt.NDArray[np.int32]]

    video_frame_patches: list[npt.NDArray[np.float32]] = field(
        default_factory=list
    )
    """Unpadded per-frame pixel values, flat across all videos.
    Each entry is ``[n_real_patches, patch_dim]``."""

    video_frame_pos_ids: list[npt.NDArray[np.int32]] = field(
        default_factory=list
    )
    """Unpadded per-frame position IDs, flat across all videos.
    Each entry is ``[n_real_patches, 2]``."""

    video_frame_patch_counts: list[int] = field(default_factory=list)
    """Real (non-padding) patch count per frame, flat across all videos."""

    video_frame_soft_token_counts: list[int] = field(default_factory=list)
    """Output soft-token count per frame (= real_patches // k²),
    flat across all videos."""

    video_token_ranges: list[tuple[int, int]] = field(default_factory=list)
    """``(start_idx, end_idx)`` of video placeholder tokens in the prompt."""

    @property
    def needs_video_encoding(self) -> bool:
        """Whether video encoding is still needed for this context.

        Returns ``True`` when the context contains video frames whose
        placeholder tokens have not yet been fully processed (i.e. the
        active window still covers at least one video range).  Once
        ``processed_length`` moves past every video range the encoder
        can be skipped — mirroring ``needs_vision_encoding`` for images.
        """
        if not self.video_frame_patches:
            return False
        return any(
            self.tokens.processed_length < end
            for _, end in self.video_token_ranges
        )
