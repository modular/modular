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
"""Mistral Large 4's text-and-vision context."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.context.context import TextAndVisionContext


@dataclass(kw_only=True, slots=True)
class MistralLarge4VisionContext(TextAndVisionContext):
    """A text-and-vision context whose image spans hold break tokens.

    An image's span runs from its first image token to its last, with a
    break token after every row but the last and the last row closed by
    the end token. Only the image tokens take an embedding -- the break
    and end tokens embed as ordinary text, exactly as the reference's
    placeholder mask does (``input_ids == image_token_id``) -- so the
    scatter indices are the image tokens' own positions, stored here
    rather than derived from the span.
    """

    image_token_indices: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    """Positions of the image tokens in the prompt."""
