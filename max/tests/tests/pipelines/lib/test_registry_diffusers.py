# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Tests for PipelineRegistry with diffusers models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.lib import (  # type: ignore[attr-defined]
    PixelGenerationConfig,
    SupportedEncoding,
)
from max.pipelines.lib.config_enums import RopeType
from max.pipelines.lib.registry import (
    PipelineRegistry,
    SupportedArchitecture,
)


def create_diffusers_model_structure(
    tmp_path: Path,
    pipeline_class: str = "FluxPipeline",
) -> Path:
    """Create a mock diffusers model directory structure."""
    model_index: dict[str, Any] = {
        "_class_name": pipeline_class,
        "_diffusers_version": "0.30.0",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(model_index))

    for component_name in ["transformer", "vae", "tokenizer"]:
        (tmp_path / component_name).mkdir(exist_ok=True)

    return tmp_path


def create_mock_architecture(
    name: str,
    task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> SupportedArchitecture:
    """Create a mock SupportedArchitecture for testing."""
    mock_pipeline_model = MagicMock()
    mock_pipeline_model.calculate_max_seq_len = MagicMock(return_value=2048)

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(eos=49407)

    return SupportedArchitecture(
        name=name,
        example_repo_ids=[f"test/{name.lower()}"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=mock_pipeline_model,
        task=task,
        tokenizer=mock_tokenizer,
        context_type=MagicMock(),
        default_weights_format=WeightsFormat.safetensors,
        rope_type=RopeType.none,
    )


def test_retrieve_architecture_with_diffusers_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that retrieve_architecture uses diffusers_config when available."""
    monkeypatch.setenv("MODULAR_PIPELINE_DEFER_RESOLVE", "true")
    create_diffusers_model_structure(tmp_path, pipeline_class="FluxPipeline")

    flux_arch = create_mock_architecture(
        "FluxPipeline", task=PipelineTask.PIXEL_GENERATION
    )
    registry = PipelineRegistry([flux_arch])

    config = PixelGenerationConfig(model={"model_path": str(tmp_path)})

    arch = registry.retrieve_architecture(pipeline_config=config)

    assert arch is not None
    assert arch.name == "FluxPipeline"
    assert arch.task == PipelineTask.PIXEL_GENERATION


def test_retrieve_pipeline_task_returns_pixel_generation_for_diffusers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test retrieve_pipeline_task returns PIXEL_GENERATION for diffusers."""
    monkeypatch.setenv("MODULAR_PIPELINE_DEFER_RESOLVE", "true")
    create_diffusers_model_structure(tmp_path, pipeline_class="FluxPipeline")

    flux_arch = create_mock_architecture(
        "FluxPipeline", task=PipelineTask.PIXEL_GENERATION
    )
    registry = PipelineRegistry([flux_arch])

    config = PixelGenerationConfig(model={"model_path": str(tmp_path)})

    task = registry.retrieve_pipeline_task(config)

    assert task == PipelineTask.PIXEL_GENERATION


def test_retrieve_context_type_for_diffusers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test retrieve_context_type returns correct context for diffusers models."""
    monkeypatch.setenv("MODULAR_PIPELINE_DEFER_RESOLVE", "true")
    create_diffusers_model_structure(tmp_path, pipeline_class="FluxPipeline")

    mock_context_type = MagicMock()
    flux_arch = create_mock_architecture(
        "FluxPipeline", task=PipelineTask.PIXEL_GENERATION
    )
    # Override the context_type with the mock
    flux_arch.context_type = mock_context_type

    registry = PipelineRegistry([flux_arch])

    config = PixelGenerationConfig(model={"model_path": str(tmp_path)})

    context_type = registry.retrieve_context_type(config)

    assert context_type is mock_context_type


def test_retrieve_factory_returns_tokenizer_from_architecture_for_diffusers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test retrieve_factory returns tokenizer from architecture for diffusers."""
    monkeypatch.setenv("MODULAR_PIPELINE_DEFER_RESOLVE", "true")
    create_diffusers_model_structure(tmp_path, pipeline_class="FluxPipeline")

    # Create a mock tokenizer that the architecture will provide
    expected_tokenizer = MagicMock()
    expected_tokenizer.eos = 49407

    flux_arch = create_mock_architecture(
        "FluxPipeline", task=PipelineTask.PIXEL_GENERATION
    )
    # Override the tokenizer to return our expected tokenizer
    flux_arch.tokenizer = MagicMock(return_value=expected_tokenizer)

    registry = PipelineRegistry([flux_arch])

    config = PixelGenerationConfig(model={"model_path": str(tmp_path)})

    tokenizer, factory = registry.retrieve_factory(
        config, task=PipelineTask.PIXEL_GENERATION
    )

    # Tokenizer should be the one returned by arch.tokenizer
    assert tokenizer is expected_tokenizer
    # Factory should be callable
    assert callable(factory)
    # Verify arch.tokenizer was called with expected arguments
    flux_arch.tokenizer.assert_called_once()


def test_retrieve_factory_returns_pixel_generation_pipeline_for_diffusers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test retrieve_factory returns PixelGenerationPipeline for diffusers."""
    from functools import partial

    monkeypatch.setenv("MODULAR_PIPELINE_DEFER_RESOLVE", "true")
    create_diffusers_model_structure(tmp_path, pipeline_class="FluxPipeline")

    flux_arch = create_mock_architecture(
        "FluxPipeline", task=PipelineTask.PIXEL_GENERATION
    )
    registry = PipelineRegistry([flux_arch])

    config = PixelGenerationConfig(model={"model_path": str(tmp_path)})

    _, factory = registry.retrieve_factory(
        config, task=PipelineTask.PIXEL_GENERATION
    )

    # Factory should be a partial function wrapping PixelGenerationPipeline
    assert isinstance(factory, partial)
    # The func should be PixelGenerationPipeline (with type parameter)
    assert factory.func.__name__ == "PixelGenerationPipeline"


def test_retrieve_architecture_returns_none_for_unregistered_diffusers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test retrieve_architecture returns None for unregistered diffusers arch."""
    monkeypatch.setenv("MODULAR_PIPELINE_DEFER_RESOLVE", "true")
    create_diffusers_model_structure(
        tmp_path, pipeline_class="UnregisteredPipeline"
    )

    # Register a different architecture
    flux_arch = create_mock_architecture(
        "FluxPipeline", task=PipelineTask.PIXEL_GENERATION
    )
    registry = PipelineRegistry([flux_arch])

    config = PixelGenerationConfig(model={"model_path": str(tmp_path)})

    arch = registry.retrieve_architecture(pipeline_config=config)

    assert arch is None
