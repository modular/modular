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
"""Tests for cascade pipeline dispatch in ``all_pipelines``.

Covers the two dispatch paths ``build_pipeline`` exposes: exact-match dummy
fixtures, and architecture-driven selection via
``SupportedArchitecture.cascade_pipeline_factory``. The architecture-driven
cases stub ``_resolve_architecture`` so no Hugging Face config is downloaded,
while a dedicated test exercises the real registry wiring (a text-generation
architecture declares the common text pipeline class).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from max.experimental.cascade.pipelines import all_pipelines
from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
)
from max.experimental.cascade.pipelines.dummy_imgen import DummyImageGenPipeline
from max.experimental.cascade.pipelines.dummy_textgen import (
    DummyTextGenPipeline,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest


def _config(model_path: str) -> PipelineConfig:
    """Build an unresolved config for construction-only (no-download) tests."""
    return PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path=model_path)})
    )


@pytest.mark.asyncio
async def test_build_pipeline_dummy_textgen() -> None:
    pipeline = await all_pipelines.build_pipeline(_config("dummy_textgen"))
    assert isinstance(pipeline, DummyTextGenPipeline)


@pytest.mark.asyncio
async def test_build_pipeline_dummy_imgen() -> None:
    pipeline = await all_pipelines.build_pipeline(_config("dummy_imgen"))
    assert isinstance(pipeline, DummyImageGenPipeline)


@pytest.mark.asyncio
async def test_build_pipeline_uses_arch_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolve the real Llama architecture (no HF download needed to get the
    # SupportedArchitecture object) but stub the model-path -> arch resolution
    # so the test never hits the network. The dispatcher should build the
    # architecture's declared cascade pipeline class.
    register_all_models()
    arch = PIPELINE_REGISTRY.retrieve_architecture("LlamaForCausalLM")
    assert arch is not None
    monkeypatch.setattr(
        all_pipelines, "_resolve_architecture", lambda config: arch
    )
    pipeline = await all_pipelines.build_pipeline(_config("some-org/some-llm"))
    assert isinstance(pipeline, CommonTextGenPipeline)


@pytest.mark.asyncio
async def test_build_pipeline_arch_without_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_arch = SimpleNamespace(
        name="SomeForCausalLM", cascade_pipeline_factory=None
    )
    monkeypatch.setattr(
        all_pipelines, "_resolve_architecture", lambda config: stub_arch
    )
    with pytest.raises(NotImplementedError, match="no cascade"):
        await all_pipelines.build_pipeline(_config("some-org/some-model"))


@pytest.mark.asyncio
async def test_build_pipeline_no_models() -> None:
    config = PipelineConfig(models=ModelManifest({}))
    with pytest.raises(ValueError, match="No models specified"):
        await all_pipelines.build_pipeline(config)


def test_common_textgen_construction() -> None:
    pipeline = CommonTextGenPipeline(_config("some-org/llm"))
    assert pipeline.tokenizer.model_path == "some-org/llm"


def test_llama_arch_declares_cascade_factory() -> None:
    # End-to-end check of the integration the dispatcher depends on: a real
    # text-generation architecture declares the common text pipeline class.
    register_all_models()
    arch = PIPELINE_REGISTRY.retrieve_architecture("LlamaForCausalLM")
    assert arch is not None
    assert arch.cascade_pipeline_factory is CommonTextGenPipeline
