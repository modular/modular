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

from __future__ import annotations

import pytest
from max.driver import DeviceSpec, accelerator_count
from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, TextContext
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib.registry import SupportedArchitecture
from max.pipelines.lib.tokenizer import TextTokenizer
from test_common.pipeline_model_dummy import (
    DummyLlamaArchConfig,
    DummyLlamaPipelineModel,
)
from test_common.registry import prepare_registry

pytest.mark.skip(
    reason="TODO MODELS-890: Reenable these tests when we do not call out to HuggingFace / move to HF workflow",
)


@prepare_registry
def test_registry__retrieve_architecture_with_legacy_module() -> None:
    """Test that retrieve_architecture works with use_legacy_module flag (default=True)."""
    # Register the legacy architecture (with _Legacy suffix)
    legacy_arch = SupportedArchitecture(
        name="LlamaForCausalLM_Legacy",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(legacy_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_legacy_module=True,
    )

    assert arch is legacy_arch


@prepare_registry
def test_registry__retrieve_architecture_without_legacy_module() -> None:
    """Test that retrieve_architecture falls back to legacy when new arch not registered."""
    # Only register the legacy architecture (with _Legacy suffix)
    legacy_arch = SupportedArchitecture(
        name="LlamaForCausalLM_Legacy",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(legacy_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    # When use_legacy_module=False but only legacy exists, should fall back
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_legacy_module=False,
    )

    assert arch is legacy_arch


@prepare_registry
def test_registry__retrieve_architecture_new_module() -> None:
    """Test that when use_legacy_module=False, new Module arch is chosen."""
    # Register both architectures
    # Legacy arch has _Legacy suffix
    legacy_arch = SupportedArchitecture(
        name="LlamaForCausalLM_Legacy",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(legacy_arch)

    # New arch uses the standard HF name (no suffix)
    new_arch = SupportedArchitecture(
        name="LlamaForCausalLM",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(new_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    arch_new = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_legacy_module=False,
    )

    arch_legacy = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_legacy_module=True,
    )

    assert arch_new is new_arch
    assert arch_legacy is legacy_arch


@prepare_registry
def test_config__use_legacy_module_default_is_true() -> None:
    """Test that use_legacy_module defaults to True in PipelineConfig for backward compat."""
    # Register legacy arch with _Legacy suffix (matches use_legacy_module=True)
    legacy_arch = SupportedArchitecture(
        name="LlamaForCausalLM_Legacy",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(legacy_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    assert config.use_legacy_module is True


@prepare_registry
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__use_legacy_module_can_be_set_to_false() -> None:
    """Test that use_legacy_module can be set to False in PipelineConfig."""
    # Register the new Module architecture (standard HF name, no suffix)
    new_arch = SupportedArchitecture(
        name="LlamaForCausalLM",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(new_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )

    assert config.use_legacy_module is False


@prepare_registry
def test_config__use_legacy_module_false_falls_back_to_legacy_arch() -> None:
    """Test that use_legacy_module=False falls back to legacy when no new arch registered."""
    # Only register the legacy architecture (with _Legacy suffix)
    legacy_arch = SupportedArchitecture(
        name="LlamaForCausalLM_Legacy",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
    )
    PIPELINE_REGISTRY.register(legacy_arch)

    # Should succeed by falling back to legacy arch
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            # Use only one GPU since this model does not support multi-GPU inference.
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )
    assert config.use_legacy_module is False


@prepare_registry
def test_registry__retrieve_architecture_falls_back_to_non_legacy() -> None:
    """Test that use_legacy_module=True falls back to non-legacy when only it exists."""
    # Only register non-legacy architecture (standard HF name, no suffix)
    new_arch = SupportedArchitecture(
        name="LlamaForCausalLM",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
        multi_gpu_supported=True,
    )
    PIPELINE_REGISTRY.register(new_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    # Default use_legacy_module=True, but only non-legacy exists — should fall back
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_legacy_module=True,
    )

    assert arch is new_arch


@prepare_registry
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__use_legacy_module_with_draft_model() -> None:
    """Test that use_legacy_module is respected for draft model in speculative decoding."""
    # Register the new Module architecture (standard HF name, no suffix)
    new_arch = SupportedArchitecture(
        name="LlamaForCausalLM",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding="float32",
        supported_encodings={
            "float32": ["paged"],
        },
        pipeline_model=DummyLlamaPipelineModel,
        config=DummyLlamaArchConfig,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
    )
    PIPELINE_REGISTRY.register(new_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )

    assert config.use_legacy_module is False

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_legacy_module=config.use_legacy_module,
    )

    assert arch is new_arch
