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
"""Registry wiring and verify-width resolution for MiMo-V2 with DFlash."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.mimo_v2 import memory_planner
from max.pipelines.architectures.mimo_v2.arch import mimo_v2_arch
from max.pipelines.architectures.unified_dflash_mimo_v2 import (
    MiMoV2DFlashContextModel,
    UnifiedDflashMiMoV2Model,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.memory_planner import (
    MiMoV2DFlashMemoryPlanner,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    MiMoV2DFlashContextConfig,
    UnifiedDflashMiMoV2Config,
    mimo_dflash_draft_width,
)
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.weights import HuggingFaceRepo

DRAFT_HF = {"dflash_config": {"mask_token_id": 151675, "target_layer_ids": [0]}}


def test_both_architectures_resolve_by_name() -> None:
    fused = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDflashMiMoV2ForCausalLM"
    )
    assert fused is not None
    assert fused.pipeline_model is UnifiedDflashMiMoV2Model
    assert fused.config is UnifiedDflashMiMoV2Config
    assert fused.checkpoint_draft_width is mimo_dflash_draft_width
    base_ctx = PIPELINE_REGISTRY.retrieve_architecture(
        "MiMoV2DFlashContextForCausalLM"
    )
    assert base_ctx is not None
    assert base_ctx.pipeline_model is MiMoV2DFlashContextModel
    assert base_ctx.config is MiMoV2DFlashContextConfig
    # Both serve the target checkpoint as the base arch does.
    for arch in (fused, base_ctx):
        assert arch.tokenizer is mimo_v2_arch.tokenizer
        assert arch.supported_encodings == mimo_v2_arch.supported_encodings
        assert arch.multi_gpu_supported
        assert arch.memory_planner is MiMoV2DFlashMemoryPlanner


def test_the_planner_counts_the_drafter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        memory_planner, "adapted_weights_size", lambda config, n: 1000 * n
    )
    (tmp_path / "dflash").mkdir()
    (tmp_path / "dflash" / "dflash_draft_model.safetensors").write_bytes(
        bytes(345)
    )
    model = SimpleNamespace(
        huggingface_config=object(),
        device_specs=[0, 1],
        huggingface_weight_repo=HuggingFaceRepo(str(tmp_path)),
    )
    planner = MiMoV2DFlashMemoryPlanner(
        SimpleNamespace(devices=[], get_kv_params=lambda: None)
    )
    # The base graph with the context writer reads the target's drafter.
    base_ctx = SimpleNamespace(model=model, draft_model=None)
    assert planner.estimate_weights_size(base_ctx) == 2000 + 345
    # The fused graph loads the draft model's.
    fused = SimpleNamespace(
        model=model, draft_model=SimpleNamespace(weights_size=lambda: 678)
    )
    assert planner.estimate_weights_size(fused) == 2000 + 678


def _resolved(
    speculative: SpeculativeConfig | None, draft_arch: str = "LlamaForCausalLM"
) -> str:
    models: dict[str, Any] = {
        "main": SimpleNamespace(
            huggingface_config=SimpleNamespace(
                architectures=["MiMoV2ForCausalLM"]
            )
        ),
        # The CLI path has already rewritten DFlashDraftModel by now.
        "draft": SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[draft_arch])
        ),
    }
    _apply_speculative_target_architecture(speculative, models)
    return models["main"].huggingface_config.architectures[0]


@pytest.mark.parametrize("draft_arch", ["LlamaForCausalLM", "DFlashDraftModel"])
def test_dflash_selects_the_fused_graph(draft_arch: str) -> None:
    assert (
        _resolved(SpeculativeConfig(speculative_method="dflash"), draft_arch)
        == "UnifiedDflashMiMoV2ForCausalLM"
    )


def test_other_methods_keep_the_base_graph() -> None:
    assert _resolved(None) == "MiMoV2ForCausalLM"
    assert (
        _resolved(SpeculativeConfig(speculative_method="dflash2"))
        == "MiMoV2ForCausalLM"
    )


def _width(k: int | None, block: int = 8) -> int:
    return mimo_dflash_draft_width(
        SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=k
        ),
        None,
        {**DRAFT_HF, "block_size": block},
    )


def test_verify_width_is_decoupled_from_the_block() -> None:
    assert _width(None) == 7
    for k in range(1, 8):
        assert _width(k) == k
    for k in (0, 8):
        with pytest.raises(ValueError, match="verifies 1 to 7"):
            _width(k)
