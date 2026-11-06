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
"""Maps Xiaomi's MiMo-V2.6-Flash checkpoint onto MAX's MiMo-V2 modules.

Every tensor loads as the checkpoint stores it; the model reads Xiaomi's
layouts itself. The vision, audio and MTP tensors are not loaded.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.pipelines.lib import Bitcast, Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig


def mimo_v2_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a MiMo-V2 checkpoint onto the decoder.

    The weights take the checkpoint's names without ``model.``, each FP8
    projection's ``weight_scale_inv`` as ``weight_scale`` and the router's
    ``mlp.gate.weight`` as ``mlp.gate.gate_score.weight``. The routed
    experts' ``weight_scale`` bytes are E8M0, which the checkpoint stores as
    ``U8``.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        Ignore(
            r"^(?:visual|audio_encoder|speech_embeddings)\.",
            reason="the vision and audio towers, which MAX does not implement",
        ),
        Ignore(
            r"^model\.mtp\.",
            reason="the MTP layers, which no MiMo-V2 draft loads",
        ),
        WeightRenaming(r"^model\.", ""),
        WeightRenaming(r"\.weight_scale_inv$", ".weight_scale"),
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
        Bitcast(
            r"\.mlp\.experts\..+\.weight_scale$",
            src=DType.uint8,
            dst=DType.float8_e8m0fnu,
        ),
    )
