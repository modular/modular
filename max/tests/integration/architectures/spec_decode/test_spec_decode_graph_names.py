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

"""The graph names a target declares for each spec-decode method.

Mach looks the fused graph up by name and reads its inputs by layout, so a
rename on this side breaks Mach serving with no Python test failing. Mach
hard-codes these names in ``mach/crates/engine-core/src/model_registry.rs``
and the per-model contract files beside it; change both together.
"""

from __future__ import annotations

import pytest
from max.pipelines.lib.arch_lookup import ARCH_LOOKUP

MACH_GRAPH_NAMES = {
    "DeepseekV3ForCausalLM": {
        "eagle": "eagle3_deepseekV3_graph",
        "Eagle3MHADeepseekV3DraftModel": "eagle3_mha_deepseekV3_graph",
        "mtp": "deepseekV3_with_mtp_graph",
    },
    "DeepseekV4ForCausalLM": {"dspark": "unified_dspark_deepseekV4"},
    "Gemma4ForConditionalGeneration": {"mtp": "gemma4_with_mtp_graph"},
    "KimiK25ForConditionalGeneration": {
        "Eagle3DeepseekV3DraftModel": "eagle3_kimik25_graph",
        "Eagle3MHADeepseekV3DraftModel": "eagle3_mha_kimik25_graph",
        "dflash": "unified_dflash_kimi_k25_graph",
    },
    "MiMoV2ForCausalLM": {"dflash": "unified_dflash_mimo_v2"},
    "Qwen3_5ForConditionalGeneration": {
        "dflash2": "qwen3_5_with_dflash2_graph",
        "mtp": "qwen3_5_with_mtp_graph",
    },
}


@pytest.mark.parametrize("target", sorted(MACH_GRAPH_NAMES))
def test_a_target_declares_exactly_the_graph_names_mach_expects(
    target: str,
) -> None:
    arch = ARCH_LOOKUP.find(target)
    assert arch is not None
    assert dict(arch.spec_decode_graph_names) == MACH_GRAPH_NAMES[target]
