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
"""The Pixtral context of the registered decoder and vision roles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.pipelines.context import TextAndVisionContext


@dataclass(kw_only=True)
class PixtralContext(TextAndVisionContext):
    """A Pixtral prompt, whose image spans interleave break tokens.

    An image's span runs from its first ``[IMG]`` token to its last, with an
    ``[IMG_BREAK]`` after every row of patches but the last. The encoder
    emits one row per patch, so only the ``[IMG]`` tokens take image rows.
    """

    @property
    def image_token_indices(self) -> npt.NDArray[np.int32]:
        """Positions of the ``[IMG]`` tokens, which the tokenizer lists first."""
        spans = super().image_token_indices
        return spans[self.tokens.all[spans] == self.vision_token_ids[0]]
