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
"""Guards the lazy registrations the DSpark Nemotron-H wiring depends on."""

from __future__ import annotations

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.unified_dspark_nemotron_h import (
    UnifiedDSparkNemotronHConfig,
    UnifiedDSparkNemotronHModel,
)
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.arch_lookup import find_architecture

_UNIFIED = "UnifiedDSparkNemotronHForCausalLM_ModuleV3"


def test_unified_arch_is_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture(_UNIFIED)
    assert arch is not None
    assert arch.pipeline_model is UnifiedDSparkNemotronHModel
    assert arch.config is UnifiedDSparkNemotronHConfig
    assert arch.checkpoint_draft_width is not None


def test_config_rewrite_resolves_the_unified_arch() -> None:
    # The rewrite names the architecture without the ModuleV3 suffix.
    for prefer_module_v3 in (False, True):
        arch = find_architecture(
            "UnifiedDSparkNemotronHForCausalLM",
            prefer_module_v3=prefer_module_v3,
        )
        assert arch is not None and arch.name == _UNIFIED


def test_mixed_batches_advance_draft_less() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture(_UNIFIED)
    assert arch is not None
    assert arch.supports_spec_decode_mixed_batches is False


def test_draft_arch_is_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("Qwen3DSparkModel")
    assert arch is not None
    assert "float4_e2m1fnx2" in arch.supported_encodings


def test_the_unified_graph_replaces_the_decoder() -> None:
    # Otherwise the base compiles the plain Nemotron-H decoder.
    assert (
        UnifiedDSparkNemotronHModel._load_decoder
        is not ModuleV3PipelineModelWithKVCache._load_decoder
    )
