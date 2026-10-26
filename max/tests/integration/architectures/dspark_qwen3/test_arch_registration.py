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
"""Pins how a Nemotron-H target and its DSpark draft pair."""

from __future__ import annotations

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    spec_decode_architecture,
)


def test_draft_arch_is_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("Qwen3DSparkModel")
    assert arch is not None
    assert arch.pipeline_model is None
    assert "float4_e2m1fnx2" in arch.supported_encodings


def test_pairs_with_the_modulev3_target() -> None:
    # Nemotron-H builds its module only as ModuleV3.
    arch = spec_decode_architecture(
        "NemotronHForCausalLM", "dspark", "Qwen3DSparkModel"
    )
    assert arch is not None
    assert arch.name == "NemotronHForCausalLM_ModuleV3+dspark"
    assert arch.checkpoint_draft_width is not None


def test_mixed_batches_advance_draft_less() -> None:
    arch = spec_decode_architecture(
        "NemotronHForCausalLM", "dspark", "Qwen3DSparkModel"
    )
    assert arch is not None
    assert arch.supports_spec_decode_mixed_batches is False


def test_keeps_prefix_caching_on() -> None:
    arch = spec_decode_architecture(
        "NemotronHForCausalLM", "dspark", "Qwen3DSparkModel"
    )
    assert arch is not None
    assert "enable_prefix_caching" not in arch.required_arguments
