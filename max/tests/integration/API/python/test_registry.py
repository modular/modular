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

from typing import Any

import pytest
from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, TextContext
from max.pipelines.lib.config_enums import RopeType, SupportedEncoding
from max.pipelines.lib.registry import SupportedArchitecture
from max.pipelines.lib.tokenizer import TextTokenizer
from test_common.mocks import mock_pipeline_config_hf_dependencies
from test_common.pipeline_model_dummy import (
    DUMMY_GEMMA_ARCH,
    DUMMY_LLAMA_ARCH,
    DummyPipelineModel,
)
from test_common.registry import prepare_registry


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_register() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    assert "LlamaForCausalLM" in PIPELINE_REGISTRY.architectures

    # This should fail when registering the architecture for a second time.
    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_max_engine() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    with pytest.raises(ValueError):
        config = PipelineConfig(
            model_path="GSAI-ML/LLaDA-8B-Instruct",
            # This forces it to fail if we dont have it.
            max_batch_size=1,
            max_length=1,
            trust_remote_code=True,
        )


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine() -> (
    None
):
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    # Should now raise an error since HuggingFace fallback is removed
    with pytest.raises(
        ValueError, match="MAX-optimized architecture not available"
    ):
        config = PipelineConfig(
            model_path="GSAI-ML/LLaDA-8B-Instruct",
            max_batch_size=1,
            max_length=1,
            trust_remote_code=True,
        )

    @prepare_registry
    @mock_pipeline_config_hf_dependencies
    def test_registry__retrieve_pipeline_task_returns_text_generation() -> None:
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
        config = PipelineConfig(
            model_path="some-model",
            max_batch_size=1,
            max_length=1,
            trust_remote_code=True,
        )
        task = PIPELINE_REGISTRY.retrieve_pipeline_task(config)
        assert task == PipelineTask.TEXT_GENERATION


def test_supported_architecture__eq__method() -> None:
    """Test the __eq__ method of SupportedArchitecture class comprehensively."""

    # Create a simple weight adapter function that can be compared
    def simple_adapter(x: Any) -> Any:
        return x

    # Create two identical architectures
    arch1 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
            SupportedEncoding.q4_k: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        rope_type=RopeType.normal,
        weight_adapters={
            WeightsFormat.safetensors: simple_adapter,
            WeightsFormat.gguf: simple_adapter,
        },
        multi_gpu_supported=True,
        required_arguments={"enable_prefix_caching": False},
    )

    arch2 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
            SupportedEncoding.q4_k: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        rope_type=RopeType.normal,
        weight_adapters={
            WeightsFormat.safetensors: simple_adapter,
            WeightsFormat.gguf: simple_adapter,
        },
        multi_gpu_supported=True,
        required_arguments={"enable_prefix_caching": False},
    )

    # Test equality with identical objects
    assert arch1 == arch2
    assert arch2 == arch1

    # Test equality with self
    assert arch1 == arch1

    # Test inequality with different class
    assert arch1 != "not an architecture"
    assert arch1 != 42
    assert arch1 is not None

    # Test inequality with different field values
    arch3 = SupportedArchitecture(
        name="DifferentModel",  # Different name
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch3

    # Test inequality with different example_repo_ids
    arch4 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["different/repo"],  # Different repo IDs
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch4

    # Test inequality with different default_encoding
    arch5 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.q4_k,  # Different encoding
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch5

    # Test inequality with different supported_encodings
    arch6 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
            # Missing q4_k encoding
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch6

    # Test inequality with different pipeline_model
    arch7 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DUMMY_GEMMA_ARCH.pipeline_model,  # Different model
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch7

    # Test inequality with different task
    arch8 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,  # Different task
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch8

    # Test inequality with different tokenizer
    arch9 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=DUMMY_GEMMA_ARCH.tokenizer,  # Different tokenizer
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch9

    # Test inequality with different default_weights_format
    arch10 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,  # Different format
    )
    assert arch1 != arch10

    # Test inequality with different rope_type
    arch11 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        rope_type=RopeType.none,  # Different rope type
    )
    assert arch1 != arch11

    # Test inequality with different weight_adapters
    def different_adapter(x: Any) -> Any:
        return x + 1  # Different function

    arch12 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        weight_adapters={
            WeightsFormat.safetensors: different_adapter,  # Different weight adapters
        },
    )
    assert arch1 != arch12

    # Test inequality with different multi_gpu_supported
    arch13 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        multi_gpu_supported=False,  # Different multi_gpu_supported
    )
    assert arch1 != arch13

    arch14 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        required_arguments={"enable_prefix_caching": False},
    )
    assert arch1 != arch14

    # Test with None weight_adapters (should default to empty dict)
    arch15 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )

    arch16 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        },
        pipeline_model=DummyPipelineModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        weight_adapters={},  # Empty dict
    )
    assert arch15 == arch16
