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
"""Names Inkling 2 checkpoint tensors as the decoder loads them."""

from collections.abc import Sequence

from max.dtype import DType
from max.pipelines.lib.weight_transform import (
    Bitcast,
    Ignore,
    WeightRenaming,
    WeightRule,
)
from transformers import PretrainedConfig

from .vision import INKLING2_VISION_WEIGHT_RULES


def inkling2_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules for Inkling 2's checkpoint tensors.

    The module trees use the checkpoint's own names below ``model.llm.`` and
    ``model.visual.``, so only the prefixes and the raw ue8m0 scale bytes need
    a rule.
    """
    del huggingface_config
    return (
        # The drafter ships in its own file, and its tensor names carry no
        # `model.` prefix, so they would collide with the decoder's `norm`.
        Ignore(
            r"^(fuse_weight|mask_embed|markov_head\.|norm\.weight$|layers\.)",
            reason="the MTP drafter loads as its own draft",
        ),
        WeightRenaming(r"^model\.llm\.", ""),
        *INKLING2_VISION_WEIGHT_RULES,
        # MXFP8 scales are stored as raw uint8 ue8m0 bytes; NVFP4's are
        # already float8_e4m3fn, which the rule leaves alone.
        Bitcast(
            r"\.experts\.w(13|2)_weight_scale$",
            DType.uint8,
            DType.float8_e8m0fnu,
        ),
    )
