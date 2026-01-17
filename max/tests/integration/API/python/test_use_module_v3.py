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

from __future__ import annotations

import pytest
from max.driver import accelerator_count
from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, TextContext
from max.pipelines.lib.config_enums import SupportedEncoding
from max.pipelines.lib.registry import SupportedArchitecture
from max.pipelines.lib.tokenizer import TextTokenizer
from test_common.mocks import mock_pipeline_config_hf_dependencies
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)
from test_common.registry import prepare_registry

pytest.mark.skip(
    reason="TODO MODELS-890: Reenable these tests when we do not call out to HuggingFace / move to HF workflow",
)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__retrieve_architecture_without_module_v3() -> None:
    """Test that retrieve_architecture works without use_module_v3 flag."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        max_batch_size=1,
        max_length=128,
    )

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_module_v3=False,
    )

    assert arch is DUMMY_LLAMA_ARCH


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__retrieve_architecture_with_module_v3_not_registered() -> (
    None
):
    """Test that retrieve_architecture returns None when ModuleV3 arch not registered."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.float32,
        max_batch_size=1,
        max_length=128,
    )

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_module_v3=True,
    )

    assert arch is None


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__retrieve_architecture_with_module_v3() -> None:
    """Test that when use_module_v3=True, ModuleV3 arch is chosen over base."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    llama_v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding=SupportedEncoding.q4_0,
        supported_encodings={
            SupportedEncoding.q4_0: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyLlamaPipelineModel,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
    )
    PIPELINE_REGISTRY.register(llama_v3_arch)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        max_batch_size=1,
        max_length=128,
    )

    arch_v3 = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_module_v3=True,
    )

    arch_base = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_module_v3=False,
    )

    assert arch_v3 is llama_v3_arch
    assert arch_base is DUMMY_LLAMA_ARCH


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_config__use_module_v3_default_is_false() -> None:
    """Test that use_module_v3 defaults to False in PipelineConfig."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.float32,
        max_batch_size=1,
        max_length=128,
    )

    assert config.use_module_v3 is False


@prepare_registry
@mock_pipeline_config_hf_dependencies
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__use_module_v3_can_be_set_to_true() -> None:
    """Test that use_module_v3 can be set to True in PipelineConfig."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    llama_v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding=SupportedEncoding.float32,
        supported_encodings={
            SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyLlamaPipelineModel,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
    )
    PIPELINE_REGISTRY.register(llama_v3_arch)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.float32,
        max_batch_size=1,
        max_length=128,
        use_module_v3=True,
    )

    assert config.use_module_v3 is True


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_config__use_module_v3_fails_gracefully_without_v3_arch() -> None:
    """Test that using use_module_v3 without registered v3 arch produces appropriate error."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    with pytest.raises(
        ValueError, match="MAX-optimized architecture not available"
    ):
        PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.float32,
            max_batch_size=1,
            max_length=128,
            use_module_v3=True,
        )


@prepare_registry
@mock_pipeline_config_hf_dependencies
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__use_module_v3_with_draft_model() -> None:
    """Test that use_module_v3 is respected for draft model in speculative decoding."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    llama_v3_arch = SupportedArchitecture(
        name="LlamaForCausalLM_ModuleV3",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["trl-internal-testing/tiny-random-LlamaForCausalLM"],
        default_encoding=SupportedEncoding.float32,
        supported_encodings={
            SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyLlamaPipelineModel,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,
    )
    PIPELINE_REGISTRY.register(llama_v3_arch)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.float32,
        max_batch_size=1,
        max_length=128,
        use_module_v3=True,
    )

    assert config.use_module_v3 is True

    arch = PIPELINE_REGISTRY.retrieve_architecture(
        huggingface_repo=config.model.huggingface_model_repo,
        use_module_v3=config.use_module_v3,
    )

    assert arch is llama_v3_arch
