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
"""Rules that map a Nemotron-H checkpoint onto the ModuleV3 module tree.

The module tree mirrors the checkpoint's and declares each tensor as the
checkpoint stores it, so no rule renames or relabels a weight: the rules
only drop the tensors no graph reads. Everything else a kernel or tensor
parallelism needs, the graph does at init: it stacks each MoE mixer's
routed experts for the grouped matmul, casts the router gates to the
router's dtype and reshapes each depthwise conv weight to the
``[dim, kernel]`` the conv kernel reads.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import Ignore, WeightRule
from transformers import PretrainedConfig


def nemotron_h_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Nemotron-H checkpoint onto the decoder.

    Args:
        huggingface_config: The checkpoint's config. The dropped tensors are
            the same in every Nemotron-H checkpoint, so no rule reads it.
    """
    del huggingface_config
    return (
        Ignore(
            r"^mtp\.",
            reason="the multi-token prediction head, which no MAX graph "
            "builds: the DSpark draft reads the target's weights",
        ),
        Ignore(
            r"\.k_proj\.k_scale$",
            reason="the FP8 KV-cache scale, which no KV cache applies: the "
            "cache is BF16 or unscaled FP8",
        ),
        Ignore(
            r"\.v_proj\.v_scale$",
            reason="the FP8 KV-cache scale, which no KV cache applies: the "
            "cache is BF16 or unscaled FP8",
        ),
    )
