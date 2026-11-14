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
"""Rules that map Xiaomi's MiMo-V2 checkpoint onto the ModuleV3 decoder.

Every tensor loads as the checkpoint stores it; the model reads Xiaomi's
layouts itself. The rules only rename, relabel the routed experts' ``U8``
block scales as the E8M0 bytes they are, and drop what the text model
doesn't read.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.pipelines.lib.weight_transform import (
    Bitcast,
    Ignore,
    WeightRenaming,
    WeightRule,
)
from transformers import PretrainedConfig

_OTHER_TOWERS = "the vision and audio towers, which MAX does not implement"


def mimo_v2_weight_rules(
    huggingface_config: PretrainedConfig | None = None,
) -> Sequence[WeightRule]:
    """Returns the rules that map a MiMo-V2 checkpoint onto the decoder.

    Args:
        huggingface_config: The checkpoint's config. The names are the same
            in every MiMo-V2 checkpoint, so no rule reads it.
    """
    del huggingface_config
    return (
        Ignore(r"^visual\.", reason=_OTHER_TOWERS),
        Ignore(r"^audio_encoder\.", reason=_OTHER_TOWERS),
        Ignore(r"^speech_embeddings\.", reason=_OTHER_TOWERS),
        Ignore(
            r"^model\.mtp\.",
            reason="the multi-token prediction layers, which no MAX graph"
            " builds",
        ),
        # The repo ships the DFlash drafter in dflash/ and the audio
        # tokenizer in audio_tokenizer/. MAX's weight discovery globs both in
        # (Mach reads only the top level), and the drafter's bare names
        # would collide with the decoder's once ``model.`` is stripped. The
        # drafter loads from dflash/ as a draft checkpoint of its own.
        Ignore(
            r"^(fc|hidden_norm|norm|layers)\.",
            reason="the DFlash drafter in dflash/, which loads as its own"
            " checkpoint",
        ),
        Ignore(r"^(encoder|decoder)\.", reason="the audio tokenizer"),
        WeightRenaming(r"^model\.", ""),
        WeightRenaming(r"\.weight_scale_inv$", ".weight_scale"),
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
        Bitcast(
            r"\.mlp\.experts\.\d+\.\w+\.weight_scale$",
            src=DType.uint8,
            dst=DType.float8_e8m0fnu,
        ),
    )
