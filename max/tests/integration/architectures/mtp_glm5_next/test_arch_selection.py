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
"""How spec decoding pairs a GLM-5.3-Flash target with its MTP layer."""

from __future__ import annotations

import pytest
from max.pipelines.architectures import register_all_models
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeVisionCachePipelineModel,
    spec_decode_architecture,
)
from max.pipelines.speculative.config import SpeculativeMethod

_TARGET = "Glm5NextForConditionalGeneration"
_DRAFT = "MTPGlm5NextDraftModel"


@pytest.fixture(autouse=True, scope="module")
def _registered() -> None:
    register_all_models()


def test_mtp_pairs_the_target_with_the_layer_its_checkpoint_ships() -> None:
    fused = spec_decode_architecture(_TARGET, "mtp", None)
    assert fused is not None
    assert fused.name == f"{_TARGET}+mtp"
    # The target's batching encodes its images through the vision cache.
    assert fused.pipeline_model is _SpecDecodeVisionCachePipelineModel


def test_the_pairing_is_captured_as_a_device_graph() -> None:
    """Checks the fused step replays under capture, as the target's does."""
    fused = spec_decode_architecture(_TARGET, "mtp", None)
    assert fused is not None
    assert fused.supports_device_graph_capture


@pytest.mark.parametrize("method", ["eagle", "dflash"])
def test_a_method_without_a_checkpoint_draft_does_not_pair(
    method: SpeculativeMethod,
) -> None:
    assert spec_decode_architecture(_TARGET, method, None) is None


def test_the_draft_is_never_a_target() -> None:
    assert spec_decode_architecture(_DRAFT, "mtp", None) is None
