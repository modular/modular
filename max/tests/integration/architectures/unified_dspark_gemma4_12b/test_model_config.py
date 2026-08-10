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
"""Draft-width resolution for the unified DSpark Gemma4 config.

DSpark drafts at every block position, so the trained width is
``block_size`` itself. It is resolved as a plain int and threaded to KV
sizing and module construction without ever writing back to (or copying)
the caller's config, so a shared or frozen pipeline config is never
mutated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import pytest
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_12b.dspark_gemma4 import (
    DSparkGemma4DraftConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_12b.model_config import (
    UnifiedDSparkGemma4_12BConfig,
    resolve_dspark_num_speculative_tokens,
)
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.model_manifest import ModelManifest

BLOCK_SIZE = 7


def _make_pipeline_config(
    num_speculative_tokens: int | None,
    *,
    draft_block_size: int | None = BLOCK_SIZE,
) -> PipelineConfig:
    """Builds a minimal two-model PipelineConfig without full validation."""
    draft_hf = (
        SimpleNamespace(block_size=draft_block_size)
        if draft_block_size is not None
        else SimpleNamespace()
    )
    model_config = MAXModelConfig.model_construct(model_path="fake/target")
    draft_config = MAXModelConfig.model_construct(model_path="fake/draft")
    draft_config._huggingface_config = draft_hf
    return PipelineConfig.model_construct(
        models=ModelManifest({"main": model_config, "draft": draft_config}),
        speculative=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )


def test_resolve_returns_trained_width_for_unset() -> None:
    pipeline_config = _make_pipeline_config(None)
    original_speculative = pipeline_config.speculative

    resolved = resolve_dspark_num_speculative_tokens(pipeline_config)

    assert resolved == BLOCK_SIZE
    # The caller's config objects are never written to.
    assert pipeline_config.speculative is original_speculative
    assert original_speculative is not None
    assert original_speculative.num_speculative_tokens is None


def test_resolve_keeps_matching_value_unchanged() -> None:
    pipeline_config = _make_pipeline_config(BLOCK_SIZE)
    assert resolve_dspark_num_speculative_tokens(pipeline_config) == BLOCK_SIZE
    assert pipeline_config.speculative is not None
    assert pipeline_config.speculative.num_speculative_tokens == BLOCK_SIZE


def test_resolve_overrides_mismatch_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline_config = _make_pipeline_config(4)

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        resolved = resolve_dspark_num_speculative_tokens(pipeline_config)

    assert "overridden from 4 to 7" in caplog.text
    assert resolved == BLOCK_SIZE
    assert pipeline_config.speculative is not None
    assert pipeline_config.speculative.num_speculative_tokens == 4


def test_resolve_without_block_size_requires_explicit_value() -> None:
    with pytest.raises(ValueError, match="declares no block_size"):
        resolve_dspark_num_speculative_tokens(
            _make_pipeline_config(None, draft_block_size=None)
        )

    explicit = _make_pipeline_config(5, draft_block_size=None)
    assert resolve_dspark_num_speculative_tokens(explicit) == 5
    assert explicit.speculative is not None
    assert explicit.speculative.num_speculative_tokens == 5


@dataclass
class _FakeTextConfig:
    vocab_size: int = 1000
    num_hidden_layers: int = 30
    return_logits: object = None
    return_hidden_states: object = None
    target_layer_ids: list[int] = field(default_factory=list)


@dataclass
class _FakeTargetConfig:
    text_config: _FakeTextConfig
    devices: list[DeviceRef]


@dataclass
class _FakeDraftConfig:
    vocab_size: int = 1000


def _make_arch_config(
    speculative_config: SpeculativeConfig,
) -> UnifiedDSparkGemma4_12BConfig:
    return UnifiedDSparkGemma4_12BConfig(
        target=cast(
            Gemma4ForConditionalGenerationConfig,
            _FakeTargetConfig(_FakeTextConfig(), [DeviceRef.GPU()]),
        ),
        draft=cast(DSparkGemma4DraftConfig, _FakeDraftConfig()),
        draft_kv_params=cast(KVCacheParams, SimpleNamespace()),
        speculative_config=speculative_config,
        target_layer_ids=[10, 20],
        mask_token_id=3,
        block_size=BLOCK_SIZE,
    )


def test_validate_never_mutates_speculative_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unset = SpeculativeConfig(speculative_method="dflash")
    config = _make_arch_config(unset)
    config.validate_dspark_fields()
    assert unset.num_speculative_tokens is None

    mismatched = SpeculativeConfig(
        speculative_method="dflash", num_speculative_tokens=4
    )
    config = _make_arch_config(mismatched)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        config.validate_dspark_fields()
    assert "overridden from 4 to 7" in caplog.text
    assert mismatched.num_speculative_tokens == 4

    # Module construction reads the trained width off the arch config
    # regardless of the CLI value.
    assert config.resolve_block_size() == BLOCK_SIZE
