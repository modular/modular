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

import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from max.driver import DeviceSpec, accelerator_count
from max.dtype import DType
from max.entrypoints.cli.config import parse_task_flags
from max.interfaces import SamplingParamsGenerationConfigDefaults
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    MAXModelConfig,
    PipelineConfig,
    SamplingConfig,
)
from max.pipelines.lib.config import AudioGenerationConfig
from test_common.mocks import (
    mock_estimate_memory_footprint,
    mock_pipeline_config_resolve,
)
from test_common.pipeline_model_dummy import DUMMY_GEMMA_ARCH, DUMMY_LLAMA_ARCH
from test_common.registry import prepare_registry

# ===----------------------------------------------------------------------=== #
# Tests for utility methods
# ===----------------------------------------------------------------------=== #


class TestClickFlagParsing:
    """Test suite for the click flag parsing."""

    def test_parse_task_flags(self) -> None:
        """Test parsing of task flags."""
        flags = parse_task_flags(("flag1=value1", "flag2=value2"))
        assert flags == {"flag1": "value1", "flag2": "value2"}

    def test_parse_task_flags_with_dash_prefix(self) -> None:
        """Test parsing of task flags with dash prefix."""
        with pytest.raises(
            ValueError,
            match="Flag must be in format 'flag_name=flag_value', got: --flag3=value3",
        ):
            parse_task_flags(("flag1=value1", "flag2=value2", "--flag3=value3"))

    def test_parse_task_flags_with_space_in_value(self) -> None:
        """Test parsing of task flags with space in value."""
        with pytest.raises(
            ValueError,
            match="Flag must be in format 'flag_name=flag_value', got: flag3 value3",
        ):
            parse_task_flags(("flag1=value1", "flag2=value2", "flag3 value3"))

    def test_parse_task_flags_with_dash_in_flag_name(self) -> None:
        """Test parsing of task flags with dash in flag name."""

        # flag-3 is converted to flag_3
        flags = parse_task_flags(
            ("flag1=value1", "flag2=value2", "flag-3=value3")
        )
        assert flags == {
            "flag1": "value1",
            "flag2": "value2",
            "flag_3": "value3",
        }


class TestPipelineConfigUtilityMethods:
    """Test suite for the refactored utility methods in PipelineConfig."""

    @mock_pipeline_config_resolve
    def test_extract_kwargs_for_config_basic(self) -> None:
        """Test basic kwargs extraction for a config class."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        # Test extracting SamplingConfig kwargs
        kwargs = {
            "enable_structured_output": True,
            "enable_penalties": True,
            "enable_min_tokens": True,
            "unrelated_param": "value",
        }

        extracted = PipelineConfig._extract_kwargs_for_config(
            kwargs, SamplingConfig
        )

        # Should extract sampling-related kwargs
        assert "enable_structured_output" in extracted
        assert "enable_penalties" in extracted
        assert "enable_min_tokens" in extracted
        assert extracted["enable_structured_output"] is True
        assert extracted["enable_penalties"] is True
        assert extracted["enable_min_tokens"] is True

        # Should not extract unrelated params
        assert "unrelated_param" not in extracted

        # Original kwargs should have extracted items removed
        assert "enable_structured_output" not in kwargs
        assert "enable_penalties" not in kwargs
        assert "enable_min_tokens" not in kwargs
        assert "unrelated_param" in kwargs

    @mock_pipeline_config_resolve
    def test_extract_kwargs_for_config_with_prefix(self) -> None:
        """Test kwargs extraction with prefix filtering."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        # Test extracting with draft_ prefix
        kwargs = {
            "draft_model_path": "/path/to/draft",
            "draft_quantization_encoding": "float32",
            "model_path": "/path/to/main",
            "temperature": 0.8,
        }

        extracted = PipelineConfig._extract_kwargs_for_config(
            kwargs, MAXModelConfig, key_prefix="draft_", strip_prefix=True
        )

        # Should extract draft-prefixed kwargs with prefix stripped
        assert "model_path" in extracted
        assert "quantization_encoding" in extracted
        assert extracted["model_path"] == "/path/to/draft"
        assert extracted["quantization_encoding"] == "float32"

        # Should not extract non-prefixed items or unrelated items
        assert "temperature" not in extracted

        # Original kwargs should have draft items removed but others remain
        assert "draft_model_path" not in kwargs
        assert "draft_quantization_encoding" not in kwargs
        assert "model_path" in kwargs  # Non-prefixed should remain
        assert "temperature" in kwargs

    @mock_pipeline_config_resolve
    def test_extract_kwargs_for_config_empty_result(self) -> None:
        """Test extraction when no matching kwargs exist."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        kwargs = {
            "unrelated_param1": "value1",
            "unrelated_param2": "value2",
        }

        extracted = PipelineConfig._extract_kwargs_for_config(
            kwargs, SamplingConfig
        )

        # Should return empty dict when no matches
        assert extracted == {}

        # Original kwargs should be unchanged
        assert len(kwargs) == 2
        assert "unrelated_param1" in kwargs
        assert "unrelated_param2" in kwargs

    @mock_pipeline_config_resolve
    def test_create_lora_config_if_needed_with_lora_paths(self) -> None:
        """Test LoRA config creation when lora_paths are provided."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        kwargs = {
            "enable_lora": True,
            "lora_paths": ["/path/to/lora1", "/path/to/lora2"],
            "max_lora_rank": 32,
            "other_param": "value",
        }

        config._create_lora_config_if_needed(kwargs)

        # Should create LoRA config
        assert config.lora is not None
        assert config.lora.lora_paths == [
            "/path/to/lora1",
            "/path/to/lora2",
        ]
        assert config.lora.max_lora_rank == 32

        # Should remove LoRA-related kwargs
        assert "lora_paths" not in kwargs
        assert "max_lora_rank" not in kwargs
        assert "other_param" in kwargs  # Non-LoRA params should remain

    @mock_pipeline_config_resolve
    def test_create_lora_config_if_needed_error_on_incomplete_config(
        self,
    ) -> None:
        """Test error when LoRA config detected but no lora_paths provided."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        kwargs = {
            "max_lora_rank": 32,
            "max_num_loras": 10,
        }
        config._create_lora_config_if_needed(kwargs)
        # LoRA config should not be created if no lora_paths are provided.
        assert config.lora is None

    @mock_pipeline_config_resolve
    def test_create_draft_model_config_if_needed_with_model_path(self) -> None:
        """Test draft model config creation when model_path is provided."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        kwargs = {
            "draft_model_path": "/path/to/draft",
            "draft_quantization_encoding": "float32",
            "other_param": "value",
        }

        config._create_draft_model_config_if_needed(kwargs)

        # Should create draft model config
        assert config.draft_model is not None
        assert config.draft_model.model_path == "/path/to/draft"
        assert config.draft_model.quantization_encoding == "float32"

        # Should remove draft-related kwargs
        assert "draft_model_path" not in kwargs
        assert "draft_quantization_encoding" not in kwargs
        assert "other_param" in kwargs  # Non-draft params should remain

    @mock_pipeline_config_resolve
    def test_create_draft_model_config_if_needed_error_on_incomplete_config(
        self,
    ) -> None:
        """Test error when draft model config detected but no model_path provided."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        kwargs = {
            "draft_quantization_encoding": "float32",
            "draft_max_length": 1024,
        }

        config._create_draft_model_config_if_needed(kwargs)
        # Draft model config should not be created if no model_path is provided.
        assert config.draft_model is None

    @mock_pipeline_config_resolve
    def test_create_and_set_config_basic(self) -> None:
        """Test basic config creation and setting."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        matched_kwargs: dict[str, Any] = {
            "enable_structured_output": True,
            "enable_penalties": True,
        }
        kv_cache_kwargs: dict[str, Any] = {}

        config._create_and_set_config(
            "sampling", SamplingConfig, matched_kwargs, kv_cache_kwargs
        )

        # Should create and set the config
        assert config.sampling is not None
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is True

    @mock_pipeline_config_resolve
    def test_create_and_set_config_model_config_with_kv_cache(self) -> None:
        """Test model config creation with KV cache kwargs."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        matched_kwargs: dict[str, Any] = {"model_path": "/test/path"}
        kv_cache_kwargs: dict[str, Any] = {"kv_cache_page_size": 256}

        config._create_and_set_config(
            "model", MAXModelConfig, matched_kwargs, kv_cache_kwargs
        )

        # Should create model config with KV cache config
        assert config.model is not None
        assert config.model.model_path == "/test/path"
        assert config.model.kv_cache.kv_cache_page_size == 256

    @mock_pipeline_config_resolve
    def test_create_and_set_config_sampling_with_echo_enabled(self) -> None:
        """Test sampling config creation with echo enabled sets variable logits."""
        config = PipelineConfig(
            model=MAXModelConfig(model_path="test/model"), enable_echo=True
        )

        matched_kwargs = {"enable_min_tokens": True}
        kv_cache_kwargs: dict[str, Any] = {}

        config._create_and_set_config(
            "sampling", SamplingConfig, matched_kwargs, kv_cache_kwargs
        )

        # Should create sampling config with variable logits enabled
        assert config.sampling is not None
        assert config.sampling.enable_min_tokens is True
        assert config.sampling.enable_variable_logits is True

    @mock_pipeline_config_resolve
    def test_process_remaining_config_classes(self) -> None:
        """Test processing of remaining config classes."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        unmatched_kwargs = {
            "enable_structured_output": True,  # SamplingConfig
            "enable_penalties": True,  # SamplingConfig
            "model_path": "/override/path",  # MAXModelConfig
            "kv_cache_page_size": 128,  # KVCacheConfig
            "kv_cache_format": "float8_e4m3fn",  # KVCacheConfig
            "unknown_param": "value",  # Should remain unmatched
        }

        config._process_remaining_config_classes(unmatched_kwargs)

        # Should process and remove matched kwargs
        assert "enable_structured_output" not in unmatched_kwargs
        assert "enable_penalties" not in unmatched_kwargs
        assert "model_path" not in unmatched_kwargs
        assert "kv_cache_page_size" not in unmatched_kwargs

        # Should leave unmatched kwargs
        assert "unknown_param" in unmatched_kwargs

        # Should update configs
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is True
        assert config.model.model_path == "/override/path"
        assert config.model.kv_cache.kv_cache_page_size == 128
        assert config.model.kv_cache.cache_dtype == DType.float8_e4m3fn

    @mock_pipeline_config_resolve
    def test_process_remaining_config_classes_no_matches(self) -> None:
        """Test processing when no config classes match."""
        config = PipelineConfig(model=MAXModelConfig(model_path="test/model"))

        unmatched_kwargs = {
            "unknown_param1": "value1",
            "unknown_param2": "value2",
        }
        original_kwargs = unmatched_kwargs.copy()

        config._process_remaining_config_classes(unmatched_kwargs)

        # Should leave all kwargs unchanged when no matches
        assert unmatched_kwargs == original_kwargs

    @mock_pipeline_config_resolve
    def test_integration_full_config_initialization(
        self,
    ) -> None:
        """Test full integration of all utility methods during config initialization."""
        kwargs = {
            "model_path": "test/model",
            "max_batch_size": 4,
            # LoRA config
            "enable_lora": True,
            "lora_paths": ["/lora1", "/lora2"],
            "max_lora_rank": 64,
            # Draft model config
            "draft_model_path": "/draft/model",
            "draft_quantization_encoding": "float32",
            # Sampling config
            "enable_structured_output": True,
            # Model config with KV cache
            "quantization_encoding": "bfloat16",
            "kv_cache_page_size": 512,
        }

        config = PipelineConfig(**kwargs)  # type: ignore[arg-type]

        # Should have created all configs correctly
        assert config.max_batch_size == 4

        # LoRA config
        assert config.lora is not None
        assert config.lora.lora_paths == ["/lora1", "/lora2"]
        assert config.lora.max_lora_rank == 64

        # Draft model config
        assert config.draft_model is not None
        assert config.draft_model.model_path == "/draft/model"
        assert config.draft_model.quantization_encoding == "float32"

        # Sampling config
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is False

        # Model config with KV cache
        assert config.model.quantization_encoding == "bfloat16"
        assert config.model.kv_cache.kv_cache_page_size == 512

    @mock_pipeline_config_resolve
    def test_kv_cache_config_dtype(
        self,
    ) -> None:
        """Test that the KVCache dtype is set correctly."""
        kwargs = {
            "model_path": "trl-internal-testing/tiny-random-LlamaForCausalLM",
            # Draft model config
            "draft_model_path": "/draft/model",
            "draft_quantization_encoding": "float8_e4m3fn",
            # Model config with KV cache
            "quantization_encoding": "float4_e2m1fnx2",
            "kv_cache_page_size": 512,
        }

        config = PipelineConfig(**kwargs)  # type: ignore[arg-type]
        assert config.model.quantization_encoding == "float4_e2m1fnx2"
        # The KV cache dtype initially has a default value.
        assert config.model.kv_cache.cache_dtype == DType.float32

        assert config.draft_model is not None
        assert config.draft_model.quantization_encoding == "float8_e4m3fn"
        # The draft model KV cache dtype initially has a default value.
        assert config.draft_model.kv_cache.cache_dtype == DType.float32

        config.model.set_cache_dtype_given_quantization_encoding()
        config.draft_model.set_cache_dtype_given_quantization_encoding()
        assert config.model.kv_cache.cache_dtype == DType.bfloat16
        assert config.draft_model.kv_cache.cache_dtype == DType.bfloat16


@prepare_registry
@mock_estimate_memory_footprint
def test_validate_model_path__bad_repo_provided() -> None:
    # This test requires a HF call to check that this repo is not valid.
    with pytest.raises(Exception):
        _ = PipelineConfig(
            model=MAXModelConfig(model_path="bert-base-asdfasdf"),
        )


def test_config_init__raises_with_no_model_path() -> None:
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(
            model=MAXModelConfig(weight_path=[Path("file.gguf")])
        )


@prepare_registry
def test_config_post_init__with_weight_path_but_no_model_path() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        model=MAXModelConfig(
            weight_path=[
                Path(
                    "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                )
            ],
        ),
        use_legacy_module=False,
    )

    assert config.model.model_path == "modularai/Llama-3.1-8B-Instruct-GGUF"
    assert config.model.weight_path == [Path("llama-3.1-8b-instruct-q4_0.gguf")]


@prepare_registry
@mock_estimate_memory_footprint
def test_config_post_init__other_repo_weights(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            weight_path=[
                Path(
                    "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                )
            ],
        ),
        use_legacy_module=False,
    )

    assert (
        config.model._weights_repo_id == "modularai/Llama-3.1-8B-Instruct-GGUF"
    )
    assert config.model.weight_path == [Path("llama-3.1-8b-instruct-q4_0.gguf")]


def test_config_init__reformats_with_str_weights_path(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    # We expect this to convert the string.
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=modular_ai_llama_3_1_local_path,
            weight_path=[
                Path(
                    "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                )
            ],
        ),
        use_legacy_module=False,
    )

    assert isinstance(config.model.weight_path, list)
    assert len(config.model.weight_path) == 1
    assert isinstance(config.model.weight_path[0], Path)


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
def test_validate_model_path__correct_repo_id_provided(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=modular_ai_llama_3_1_local_path,
            quantization_encoding="bfloat16",
        ),
        use_legacy_module=False,
    )

    assert config.model.model_path == modular_ai_llama_3_1_local_path


@mock_estimate_memory_footprint
def test_config__test_incompatible_quantization_encoding(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    with pytest.raises(ValueError):
        # This should raise, as q4_k != f32.
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path=llama_3_1_8b_instruct_local_path,
                quantization_encoding="q4_k",
                weight_path=[
                    Path(
                        "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                    )
                ],
                max_length=1,
            ),
            max_batch_size=1,
            use_legacy_module=False,
        )

    # This should not raise, as float32 == f32.
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding="float32",
            weight_path=[
                Path(
                    "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                )
            ],
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            max_length=1,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_quantization_encoding_with_dtype_casting(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    with pytest.raises(ValueError):
        # This should raise, as allow_safetensors_weights_fp32_bf6_bidirectional_cast defaults to False, which
        # means it will not cast the (bfloat16) quantization encoding to
        # float32.
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path=llama_3_1_8b_instruct_local_path,
                quantization_encoding="float32",
                max_length=1,
            ),
            max_batch_size=1,
            use_legacy_module=False,
        )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_quantization_encoding_with_dtype_casting2(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # This should pass, because the flag also supports casting bfloat16 weights
    # to float32.
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding="float32",
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            max_length=1,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )
    assert config.model.kv_cache.cache_dtype == DType.float32


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_quantization_encoding_with_dtype_casting3(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # This should not raise, as allow_safetensors_weights_fp32_bf6_bidirectional_cast is set to True,
    # and the quantization encoding is set to bfloat16.
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding="bfloat16",
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            max_length=1,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )
    assert config.model.kv_cache.cache_dtype == DType.bfloat16


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_quantization_encoding_with_dtype_casting4(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Test that quantization_encoding is required when allow_safetensors_weights_fp32_bf6_bidirectional_cast is True.
    with pytest.raises(
        ValueError,
        match="--quantization-encoding must be provided when --allow-safetensors-weights-fp32-bf6-bidirectional-cast is enabled",
    ):
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path="test/model",
                allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
                # Note: quantization_encoding is not provided, which should cause the error
            ),
            use_legacy_module=False,
        )


@pytest.mark.skip(
    "TODO: This test is failing due to some int vs. MagicMock mismatch"
)
@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_retrieve_factory_with_known_architecture(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=modular_ai_llama_3_1_local_path,
            quantization_encoding="bfloat16",
            max_length=1,
        ),
        max_batch_size=1,
        use_legacy_module=False,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_retrieve_factory_with_unsupported_model_path(
    gemma_3_1b_it_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Should now raise an error since HuggingFace fallback is removed
    with pytest.raises(
        ValueError, match="MAX-optimized architecture not available"
    ):
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path=gemma_3_1b_it_local_path, max_length=1
            ),
            max_batch_size=1,
            use_legacy_module=False,
        )


class LimitedPickler(pickle.Unpickler):
    """A custom Unpickler class that checks for transformer modules."""

    def find_class(self, module: str, name: str) -> type:
        if module.startswith("transformers"):
            raise AssertionError(
                "Tried to unpickle class from transformers module, raising an "
                "error because this may break in serving."
            )
        return super().find_class(module, name)


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
def test_config_is_picklable(
    tmp_path: Path, modular_ai_llama_3_1_local_path: str
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=modular_ai_llama_3_1_local_path,
            quantization_encoding="bfloat16",
        ),
        use_legacy_module=False,
    )

    config.model._huggingface_config = None

    pickle_path = tmp_path / "config.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(config, f)

    with open(pickle_path, "rb") as f:
        limited_pickler = LimitedPickler(f)
        loaded_config = limited_pickler.load()

    assert loaded_config == config


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
def test_config__validates_supported_device(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Valid device/encoding combinations.
    _ = PipelineConfig(
        model=MAXModelConfig(
            model_path=modular_ai_llama_3_1_local_path,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding="float32",
            max_length=1,
        ),
        use_legacy_module=False,
    )

    if accelerator_count() == 0:
        with pytest.raises(ValueError):
            _ = PipelineConfig(
                model=MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="float32",
                    max_length=1,
                ),
                use_legacy_module=False,
            )
    else:
        _ = PipelineConfig(
            model=MAXModelConfig(
                model_path=modular_ai_llama_3_1_local_path,
                device_specs=[DeviceSpec.accelerator()],
                quantization_encoding="bfloat16",
                max_length=1,
            ),
            use_legacy_module=False,
        )

    with pytest.raises(
        ValueError, match="not compatible with the selected device type 'cpu'"
    ):
        # Invalid device/encoding combinations.
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path=modular_ai_llama_3_1_local_path,
                device_specs=[DeviceSpec.cpu()],
                quantization_encoding="bfloat16",
                max_length=1,
            ),
            use_legacy_module=False,
        )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
def test_config__validates_lora_configuration(
    llama_3_1_8b_instruct_local_path: str, llama_3_1_8b_lora_local_path: str
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Test LoRA configuration with valid config
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            kv_cache=KVCacheConfig(enable_prefix_caching=False),
            max_length=1,
        ),
        lora=LoRAConfig(
            enable_lora=True, lora_paths=[llama_3_1_8b_lora_local_path]
        ),
        use_legacy_module=False,
    )
    assert config.lora is not None
    assert config.lora.lora_paths[0] == llama_3_1_8b_lora_local_path
    assert config.lora.max_lora_rank == 16
    assert config.lora.max_num_loras == 1


@prepare_registry
@mock_estimate_memory_footprint
def test_config__validates_lora_only_supported_for_llama(
    gemma_3_1b_it_local_path: str,
) -> None:
    """Test that LoRA validation fails for non-Llama models."""

    PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH, allow_override=True)

    # Test that enabling LoRA on a non-Llama model raises ValueError
    with pytest.raises(
        ValueError,
        match=r"LoRA is not currently supported for architecture.*LoRA support is currently only available for Llama-3\.x models",
    ):
        _ = PipelineConfig(
            model=MAXModelConfig(
                model_path=gemma_3_1b_it_local_path,
                device_specs=[DeviceSpec.accelerator()],
                quantization_encoding="bfloat16",
                kv_cache=KVCacheConfig(enable_prefix_caching=False),
                max_length=1,
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            use_legacy_module=False,
        )


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_estimate_memory_footprint
def test_config__validates_lora_works_for_llama(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that LoRA validation passes for Llama models."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            kv_cache=KVCacheConfig(enable_prefix_caching=False),
            max_length=1,
        ),
        lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
        use_legacy_module=False,
    )

    # Verify LoRA config was created successfully
    assert config.lora is not None
    assert config.lora.enable_lora is True
    assert config.lora.lora_paths == ["/some/lora/path"]


@prepare_registry
@mock_estimate_memory_footprint
def test_config__validates_lora_incompatible_with_prefix_caching(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that LoRA and prefix caching cannot be enabled together."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Test that enabling both LoRA and prefix caching raises ValueError
    with pytest.raises(
        ValueError,
        match=r"LoRA is not compatible with prefix caching\. Please disable prefix caching by using the --no-enable-prefix-caching flag\.",
    ):
        _ = PipelineConfig(
            model=MAXModelConfig(
                model_path=llama_3_1_8b_instruct_local_path,
                device_specs=[DeviceSpec.accelerator()],
                quantization_encoding="bfloat16",
                kv_cache=KVCacheConfig(enable_prefix_caching=True),
                max_length=1,
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            use_legacy_module=False,
        )


@prepare_registry
@mock_estimate_memory_footprint
@pytest.mark.skipif(
    accelerator_count() > 1, reason="Test requires single GPU or CPU"
)
def test_config__validates_lora_single_device_only(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            kv_cache=KVCacheConfig(enable_prefix_caching=False),
            max_length=1,
        ),
        lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
        use_legacy_module=False,
    )
    assert config.lora is not None
    assert config.lora.enable_lora is True


@pytest.mark.skip(
    reason="PAQ-1936: Failing due to unfetchable safetensors weights"
)
@prepare_registry
@mock_estimate_memory_footprint
@pytest.mark.skipif(
    accelerator_count() < 2, reason="Test requires multiple GPUs"
)
def test_config__validates_lora_fails_with_multiple_devices(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    with pytest.raises(
        ValueError,
        match=r"LoRA is currently not supported with the number of devices > 1\.",
    ):
        _ = PipelineConfig(
            model=MAXModelConfig(
                model_path=llama_3_1_8b_instruct_local_path,
                device_specs=[
                    DeviceSpec.accelerator(),
                    DeviceSpec.accelerator(),
                ],
                quantization_encoding="bfloat16",
                allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
                kv_cache=KVCacheConfig(enable_prefix_caching=False),
                max_length=1,
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            use_legacy_module=False,
        )

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            device_specs=[DeviceSpec.accelerator(), DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            max_length=1,
        ),
        use_legacy_module=False,
    )
    assert config.lora is None


def test_diffusers_config_loads_for_diffusion_pipeline() -> None:
    """Test that diffusers_config is properly loaded for diffusion pipelines."""
    # Use a small, publicly accessible diffusion model for testing
    # This model is from HuggingFace's internal testing organization
    diffusion_model = "hf-internal-testing/tiny-stable-diffusion-torch"

    # Create a model config with a diffusion pipeline
    model_config = MAXModelConfig(
        model_path=diffusion_model,
        device_specs=[DeviceSpec.cpu()],
    )

    # The diffusers_config should be automatically loaded
    diffusers_config = model_config.diffusers_config

    # Verify that diffusers_config is not None (indicating this is a diffusion pipeline)
    assert diffusers_config is not None, (
        "diffusers_config should not be None for diffusion pipelines"
    )

    # Verify that the config contains expected keys for a Stable Diffusion pipeline
    # model_index.json typically contains pipeline class name and component mapping
    assert "_class_name" in diffusers_config, (
        "diffusers_config should contain _class_name"
    )

    # The pipeline class name should indicate this is a Stable Diffusion pipeline
    assert "StableDiffusion" in diffusers_config["_class_name"], (
        f"Expected StableDiffusion pipeline, got {diffusers_config['_class_name']}"
    )

    # Verify that the config contains the "components" key
    assert "components" in diffusers_config, (
        "diffusers_config should contain 'components' key"
    )

    # Verify that the components dict contains expected components
    components = diffusers_config["components"]
    expected_components = ["vae", "unet", "text_encoder"]
    for component in expected_components:
        assert component in components, (
            f"components should contain {component} component"
        )

        # Verify each component has the expected structure
        component_data = components[component]
        assert "library" in component_data, (
            f"{component} should have 'library' field"
        )
        assert "class_name" in component_data, (
            f"{component} should have 'class_name' field"
        )
        assert "config_dict" in component_data, (
            f"{component} should have 'config_dict' field"
        )

        # Print component info to see what's being downloaded
        print(
            f"Component {component}: library={component_data['library']}, "
            f"class={component_data['class_name']}, "
            f"config_dict_keys={list(component_data['config_dict'].keys()) if component_data['config_dict'] else 'empty'}"
        )


def test_diffusers_config_is_none_for_transformer_model(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that diffusers_config is None for non-diffusion models."""
    # Create a model config with a standard transformer model (not a diffusion model)
    model_config = MAXModelConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        device_specs=[DeviceSpec.cpu()],
    )

    # The diffusers_config should be None for non-diffusion models
    diffusers_config = model_config.diffusers_config

    assert diffusers_config is None, (
        "diffusers_config should be None for non-diffusion transformer models"
    )


@pytest.mark.skip(reason="SERVOPT-972, diffusers_config assertion failure")
def test_pipeline_config_with_flux_1_dev_model() -> None:
    """Test PipelineConfig instantiation with Flux.1-dev model."""
    # Flux.1-dev is a diffusion model from Black Forest Labs
    flux_model = "black-forest-labs/FLUX.1-dev"

    # Create a PipelineConfig with Flux.1-dev
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=flux_model,
            device_specs=[DeviceSpec.cpu()],
        ),
        defer_resolve=True,
    )

    # Verify that the config was created successfully
    assert config.model.model_path == flux_model

    # Verify that diffusers_config is loaded (since Flux is a diffusion model)
    diffusers_config = config.model.diffusers_config
    assert diffusers_config is not None, (
        "diffusers_config should not be None for Flux.1-dev diffusion model"
    )

    # Verify that the config contains expected pipeline information
    assert "_class_name" in diffusers_config, (
        "diffusers_config should contain _class_name"
    )

    # Flux uses FluxPipeline class
    assert "Flux" in diffusers_config["_class_name"], (
        f"Expected Flux pipeline, got {diffusers_config['_class_name']}"
    )

    # Print component info to evaluate download behavior
    if "components" in diffusers_config:
        components = diffusers_config["components"]
        print(f"\nFlux model has {len(components)} components:")
        for component_name, component_data in components.items():
            config_dict_size = len(str(component_data.get("config_dict", {})))
            config_dict_keys = (
                list(component_data["config_dict"].keys())
                if component_data.get("config_dict")
                else "empty"
            )
            print(
                f"  {component_name}: library={component_data.get('library')}, "
                f"class={component_data.get('class_name')}, "
                f"config_dict_size={config_dict_size} chars, "
                f"config_dict_keys={config_dict_keys}"
            )


def test_pipeline_config_with_tiny_stable_diffusion() -> None:
    """Test PipelineConfig instantiation with tiny-stable-diffusion-torch model."""
    # Use the same model as in test_diffusers_config_loads_for_diffusion_pipeline
    # but test with PipelineConfig instead of MAXModelConfig
    diffusion_model = "hf-internal-testing/tiny-stable-diffusion-torch"

    # Create a PipelineConfig with the diffusion model
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=diffusion_model,
            device_specs=[DeviceSpec.cpu()],
        ),
        defer_resolve=True,
    )

    # Verify that the config was created successfully
    assert config.model.model_path == diffusion_model

    # Verify that diffusers_config is loaded
    diffusers_config = config.model.diffusers_config
    assert diffusers_config is not None, (
        "diffusers_config should not be None for diffusion pipelines"
    )

    # Verify that the config contains expected keys
    assert "_class_name" in diffusers_config, (
        "diffusers_config should contain _class_name"
    )

    # The pipeline class name should indicate this is a Stable Diffusion pipeline
    assert "StableDiffusion" in diffusers_config["_class_name"], (
        f"Expected StableDiffusion pipeline, got {diffusers_config['_class_name']}"
    )

    # Verify that the config contains the "components" key
    assert "components" in diffusers_config, (
        "diffusers_config should contain 'components' key"
    )

    # Verify that the components dict contains expected components
    components = diffusers_config["components"]
    expected_components = ["vae", "unet", "text_encoder"]
    for component in expected_components:
        assert component in components, (
            f"components should contain {component} component"
        )


class TestSamplingConfig:
    """Test suite for SamplingConfig."""

    def test_from_generation_config_sampling_defaults_with_repetition_penalty(
        self,
    ) -> None:
        """Test that enable_penalties is True when repetition_penalty is set to non-default value."""
        # Create sampling defaults with repetition_penalty=1.05
        sampling_defaults = SamplingParamsGenerationConfigDefaults(
            repetition_penalty=1.05
        )

        # Create SamplingConfig from the defaults
        sampling_config = (
            SamplingConfig.from_generation_config_sampling_defaults(
                sampling_defaults
            )
        )

        # Assert that enable_penalties is True
        assert sampling_config.enable_penalties is True

    def test_from_generation_config_sampling_defaults_with_default_repetition_penalty(
        self,
    ) -> None:
        """Test that enable_penalties is False when repetition_penalty is at default value."""
        # Create sampling defaults with repetition_penalty=1.0 (default)
        sampling_defaults = SamplingParamsGenerationConfigDefaults(
            repetition_penalty=1.0
        )

        # Create SamplingConfig from the defaults
        sampling_config = (
            SamplingConfig.from_generation_config_sampling_defaults(
                sampling_defaults
            )
        )

        # Assert that enable_penalties is False (since 1.0 is the default)
        assert sampling_config.enable_penalties is False

    def test_from_generation_config_sampling_defaults_without_penalties(
        self,
    ) -> None:
        """Test that enable_penalties is False when no penalty parameters are set."""
        # Create sampling defaults without any penalty parameters
        sampling_defaults = SamplingParamsGenerationConfigDefaults(
            temperature=0.7, top_k=50
        )

        # Create SamplingConfig from the defaults
        sampling_config = (
            SamplingConfig.from_generation_config_sampling_defaults(
                sampling_defaults
            )
        )

        # Assert that enable_penalties is False
        assert sampling_config.enable_penalties is False


@prepare_registry
@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__auto_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def patch_retrieve_architecture(
        arch_name: str,
    ) -> Generator[None, None, None]:
        with monkeypatch.context() as m:
            # Mock .huggingface_model_repo so that we don't reach out to HF
            m.setattr(MAXModelConfig, "huggingface_model_repo", Mock())
            # Force PIPELINE_REGISTRY.retrieve_architecture to return a custom arch
            arch = SimpleNamespace(name=arch_name)
            m.setattr(
                PIPELINE_REGISTRY,
                "retrieve_architecture",
                Mock(return_value=arch),
            )
            yield

    # Override enable_overlap_scheduler to True for Llama or Deepseek models
    for arch_name in (
        "LlamaForCausalLM_Legacy",
        "DeepseekV2ForCausalLM_Legacy",
        "DeepseekV3ForCausalLM_Legacy",
        "DeepseekV3_2ForCausalLM_Legacy",
        "DeepseekV3ForCausalLMNextN_Legacy",
    ):
        with patch_retrieve_architecture(arch_name):
            config = PipelineConfig(
                model=MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                ),
                max_num_steps=42,
            )
            config._validate_and_resolve_overlap_scheduler()
            assert config.enable_overlap_scheduler is True
            assert config.max_num_steps == 1

    # Don't override if the device is CPU
    with patch_retrieve_architecture("LlamaForCausalLM_Legacy"):
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path="test/model",
                device_specs=[DeviceSpec.cpu()],
            ),
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.enable_overlap_scheduler is False

    # Don't override if structured output is enabled
    with patch_retrieve_architecture("LlamaForCausalLM_Legacy"):
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path="test/model",
                device_specs=[DeviceSpec.accelerator()],
            ),
            sampling=SamplingConfig(enable_structured_output=True),
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.enable_overlap_scheduler is False

    # Don't override if the pipeline role is not PrefillAndDecode
    with patch_retrieve_architecture("LlamaForCausalLM_Legacy"):
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path="test/model",
                device_specs=[DeviceSpec.accelerator()],
            ),
            pipeline_role="prefill_only",
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.enable_overlap_scheduler is False

    # Don't override for other architectures
    with patch_retrieve_architecture("SomeOtherArchitecture"):
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path="test/model",
                device_specs=[DeviceSpec.accelerator()],
            ),
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.enable_overlap_scheduler is False


@prepare_registry
@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__validate() -> None:
    # Allow user to manually enable overlap scheduler
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator()],
        ),
        enable_overlap_scheduler=True,
    )
    config._validate_and_resolve_overlap_scheduler()
    assert config.enable_overlap_scheduler is True

    # Error out if user tries to enable overlap scheduler on CPU
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.cpu()],
        ),
        enable_overlap_scheduler=True,
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # Error out if device graph capture is enabled with data parallelism
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator(), DeviceSpec.accelerator()],
            data_parallel_degree=2,
        ),
        device_graph_capture=True,
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # Error out if device graph capture does not define max_batch_size.
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator()],
        ),
        device_graph_capture=True,
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # Device graph capture uses max_batch_size as the warmup capture bound.
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator()],
        ),
        device_graph_capture=True,
        max_batch_size=4,
    )
    config._validate_and_resolve_overlap_scheduler()
    assert config.enable_overlap_scheduler is True
    assert config.max_num_steps == 1

    # Error out if user tries to enable overlap scheduler without PrefillAndDecode
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator()],
        ),
        pipeline_role="prefill_only",
        enable_overlap_scheduler=True,
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # Error out if user tries to enable overlap scheduler with AudioGenerationConfig
    config = AudioGenerationConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator()],
        ),
        pipeline_role="prefill_and_decode",
        audio_decoder=Mock(),
        enable_overlap_scheduler=True,
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # Error out if user tries to enable overlap scheduler with structured output
    config = PipelineConfig(
        model=MAXModelConfig(
            model_path="test/model",
            device_specs=[DeviceSpec.accelerator()],
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        enable_overlap_scheduler=True,
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()
