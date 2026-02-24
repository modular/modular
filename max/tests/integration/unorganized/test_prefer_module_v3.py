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
def test_registry__retrieve_architecture_default() -> None:
    """Test that retrieve_architecture works with prefer_module_v3=False (default)."""
    # Register the ModuleV2 architecture (standard HF name, no suffix)
    v2_arch = SupportedArchitecture(
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
    PIPELINE_REGISTRY.register(v2_arch)

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
        prefer_module_v3=False,
    )

    assert arch is v2_arch


@prepare_registry
def test_registry__retrieve_architecture_v3_falls_back_to_v2() -> None:
    """Test that retrieve_architecture falls back to ModuleV2 when ModuleV3 not registered."""
    # Only register the ModuleV2 architecture (standard HF name)
    v2_arch = SupportedArchitecture(
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
    PIPELINE_REGISTRY.register(v2_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    # When prefer_module_v3=True but only ModuleV2 exists, should fall back
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        prefer_module_v3=True,
    )

    assert arch is v2_arch


@prepare_registry
def test_registry__retrieve_architecture_module_v3() -> None:
    """Test that when prefer_module_v3=True, ModuleV3 arch is chosen."""
    # Register both architectures
    # ModuleV2 arch uses standard HF name
    v2_arch = SupportedArchitecture(
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
    PIPELINE_REGISTRY.register(v2_arch)

    # ModuleV3 arch uses _ModuleV3 suffix
    v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
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
    PIPELINE_REGISTRY.register(v3_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    arch_v3 = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        prefer_module_v3=True,
    )

    arch_v2 = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        prefer_module_v3=False,
    )

    assert arch_v3 is v3_arch
    assert arch_v2 is v2_arch


@prepare_registry
def test_config__prefer_module_v3_default_is_false() -> None:
    """Test that prefer_module_v3 defaults to False in PipelineConfig for backward compat."""
    # Register ModuleV2 arch (matches prefer_module_v3=False)
    v2_arch = SupportedArchitecture(
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
    PIPELINE_REGISTRY.register(v2_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    assert config.prefer_module_v3 is False


@prepare_registry
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__prefer_module_v3_can_be_set_to_true() -> None:
    """Test that prefer_module_v3 can be set to True in PipelineConfig."""
    # Register the ModuleV3 architecture (with _ModuleV3 suffix)
    v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
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
    PIPELINE_REGISTRY.register(v3_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
        prefer_module_v3=True,
    )

    assert config.prefer_module_v3 is True


@prepare_registry
def test_config__prefer_module_v3_true_falls_back_to_v2_arch() -> None:
    """Test that prefer_module_v3=True falls back to ModuleV2 when no ModuleV3 registered."""
    # Only register the ModuleV2 architecture (standard HF name)
    v2_arch = SupportedArchitecture(
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
    PIPELINE_REGISTRY.register(v2_arch)

    # Should succeed by falling back to ModuleV2 arch
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            # Use only one GPU since this model does not support multi-GPU inference.
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
        prefer_module_v3=True,
    )
    assert config.prefer_module_v3 is True


@prepare_registry
def test_registry__retrieve_architecture_falls_back_to_v3() -> None:
    """Test that prefer_module_v3=False falls back to ModuleV3 when only it exists."""
    # Only register ModuleV3 architecture (with _ModuleV3 suffix)
    v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
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
    PIPELINE_REGISTRY.register(v3_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
    )

    # Default prefer_module_v3=False, but only ModuleV3 exists — should fall back
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        prefer_module_v3=False,
    )

    assert arch is v3_arch


@prepare_registry
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__prefer_module_v3_with_draft_model() -> None:
    """Test that prefer_module_v3 is respected for draft model in speculative decoding."""
    # Register the ModuleV3 architecture (with _ModuleV3 suffix)
    v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
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
    PIPELINE_REGISTRY.register(v3_arch)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding="float32",
            max_length=128,
        ),
        max_batch_size=1,
        prefer_module_v3=True,
    )

    assert config.prefer_module_v3 is True

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        prefer_module_v3=config.prefer_module_v3,
    )

    assert arch is v3_arch
