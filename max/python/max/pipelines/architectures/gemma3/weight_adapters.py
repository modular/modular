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
"""Maps Gemma 3 checkpoints onto the graph-API module tree."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import WeightRenaming, WeightRule
from transformers import PretrainedConfig


def gemma3_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Gemma 3 checkpoint onto the decoder.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        WeightRenaming(r"^model\.", "language_model."),
        WeightRenaming(r"^lm_head\.", "language_model.lm_head."),
    )
