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
"""The checkpoint-to-graph weight rules."""

from __future__ import annotations

import re
from collections.abc import Sequence

from max.dtype import DType
from max.pipelines.lib.weight_transform import (
    Bitcast,
    Ignore,
    WeightRenaming,
    WeightRule,
)
from transformers import PretrainedConfig

DECODER_PREFIX = "language_model."
"""The prefix the decoder role's weights sit under (``weight_prefixes``)."""

VISION_PREFIX = "vision."
"""The prefix the tower role's weights sit under."""

_TEXT = "model.llm."
_VISION = "model.visual."
_SKIP = {
    r"^(fuse_weight|mask_embed|markov_head\.|norm\.weight$|layers\.)": "the MTP drafter's tensors: they ship in their own file with no "
    "`model.` prefix, so they would collide with the decoder's `norm`",
}

_VISION_WEIGHT_LEAVES = ("backbone.embeddings.position_embedding",)
"""Tower leaves shipped without the `.weight` suffix their module declares."""


def inkling2_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map an Inkling2 checkpoint onto both roles."""
    del huggingface_config
    return (
        *(Ignore(pattern, reason=reason) for pattern, reason in _SKIP.items()),
        *(
            WeightRenaming(
                rf"^{re.escape(_VISION)}{re.escape(leaf)}$",
                VISION_PREFIX + leaf + ".weight",
            )
            for leaf in _VISION_WEIGHT_LEAVES
        ),
        WeightRenaming(r"^model\.visual\.", VISION_PREFIX),
        WeightRenaming(r"^model\.llm\.", DECODER_PREFIX),
        # MXFP8 scales are stored as raw uint8 ue8m0 bytes; NVFP4's are
        # already float8_e4m3fn, which the rule leaves alone.
        Bitcast(
            r"\.experts\.w(13|2)_weight_scale$",
            DType.uint8,
            DType.float8_e8m0fnu,
        ),
    )
