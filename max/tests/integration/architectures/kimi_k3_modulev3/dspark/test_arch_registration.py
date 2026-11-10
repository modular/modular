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
"""Pins how the K3 DSpark draft registers and pairs with its target."""

from __future__ import annotations

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    spec_decode_architecture,
)
from max.pipelines.speculative import spec_adapters


def test_draft_arch_is_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("K3DSparkModel")
    assert arch is not None
    assert arch.pipeline_model is None
    assert "bfloat16" in arch.supported_encodings


def _paired_arch(monkeypatch: pytest.MonkeyPatch) -> SupportedArchitecture:
    """The fused architecture of the ModuleV3 target and its DSpark draft.

    The target registers its ``DECODER``, so the pairing gate reads the
    adapter registry's key for the target's exact ModuleV3 name. The test
    registers that key -- all ``spec_decode_architecture`` reads of it --
    and the adapter module itself lands with the target's wiring.
    """
    monkeypatch.setitem(
        spec_adapters._TARGET_ADAPTERS,
        "KimiK3ForConditionalGeneration_ModuleV3",
        "",
    )
    arch = spec_decode_architecture(
        "KimiK3ForConditionalGeneration_ModuleV3", "dspark", "K3DSparkModel"
    )
    assert arch is not None
    return arch


def test_pairs_with_the_modulev3_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arch = _paired_arch(monkeypatch)
    assert arch.name == "KimiK3ForConditionalGeneration_ModuleV3+dspark"
    assert arch.checkpoint_draft_width is not None
    # The block driver commits per row, so mixed batches stay off until the
    # target's own verify proves them.
    assert arch.supports_spec_decode_mixed_batches is False


def test_keeps_prefix_caching_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The KDA recurrent state checkpoints only at a page boundary, which a
    # step's accepted drafts move past.
    arch = _paired_arch(monkeypatch)
    assert arch.required_arguments.get("enable_prefix_caching") is False
