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
"""Maps GLM-5 checkpoints onto the graph-API modules."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig

GLM5_1_RENAMES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^model\.", ""),
    WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
    WeightRenaming(r"\.weight_scale_inv$", ".weight_scale"),
)
"""Renames a GLM-5 checkpoint's weights as the graph-API modules name them."""


def glm5_1_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a GLM-5 checkpoint onto the decoder.

    The checkpoint's NextN layer, past the decoder's last one, is left out.

    Args:
        huggingface_config: The checkpoint's config.
    """
    return (
        Ignore(
            rf"^model\.layers\.{huggingface_config.num_hidden_layers}\.",
            reason="the NextN layer, which only multi-token prediction loads",
        ),
        *GLM5_1_RENAMES,
    )
