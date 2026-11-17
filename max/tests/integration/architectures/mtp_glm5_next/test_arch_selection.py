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
"""How spec decoding pairs GLM-5.3-Flash with its MTP layer."""

from __future__ import annotations

import pytest
from max.pipelines.architectures import register_all_models
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeModuleV3VisionCachePipelineModel,
    spec_decode_architecture,
)
from max.pipelines.modeling.types import InputModality

_TARGET = "Glm5NextForConditionalGeneration"


@pytest.fixture(autouse=True, scope="module")
def _registered() -> None:
    register_all_models()


def test_mtp_pairs_the_target_with_its_layer() -> None:
    fused = spec_decode_architecture(
        _TARGET, "mtp", None, prefer_module_v3=True
    )
    assert fused is not None
    assert fused.name == f"{_TARGET}_ModuleV3+mtp"
    # The target encodes images through the vision cache, and the layer
    # reads them through the target's hidden states.
    assert fused.pipeline_model is _SpecDecodeModuleV3VisionCachePipelineModel
    assert fused.input_modalities == {
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    }


def test_the_pairing_turns_prefix_caching_off() -> None:
    """Checks the KDA state's page-boundary checkpoints are never published
    at a depth the accepted drafts moved past."""
    fused = spec_decode_architecture(
        _TARGET, "mtp", None, prefer_module_v3=True
    )
    assert fused is not None
    assert fused.required_arguments["enable_prefix_caching"] is False


def test_the_pairing_needs_no_prefer_module_v3() -> None:
    """Without ``prefer_module_v3`` the pairing falls back to the only build."""
    fused = spec_decode_architecture(_TARGET, "mtp", None)
    assert fused is not None
    assert fused.name == f"{_TARGET}_ModuleV3+mtp"
