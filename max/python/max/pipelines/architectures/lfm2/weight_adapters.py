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
"""Maps LFM2 checkpoints onto the graph-API module tree."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import WeightRenaming, WeightRule
from transformers import PretrainedConfig


def lfm2_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map an LFM2 checkpoint onto the decoder.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        WeightRenaming(r"^model\.embedding_norm\.", "norm."),
        WeightRenaming(r"^model\.", ""),
        WeightRenaming(r"\.self_attn\.out_proj\.", ".self_attn.o_proj."),
        WeightRenaming(
            r"\.self_attn\.(q|k)_layernorm\.", r".self_attn.\1_norm."
        ),
        WeightRenaming(r"\.conv\.conv\.(weight|bias)$", r".conv.conv_\1"),
    )
