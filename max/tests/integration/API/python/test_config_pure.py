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

import pickle
from pathlib import Path
from typing import Any

import pytest
from max.driver import DeviceSpec, accelerator_count
from max.entrypoints.cli.config import parse_task_flags
from max.interfaces import SamplingParamsGenerationConfigDefaults
from max.pipelines import PIPELINE_REGISTRY, SupportedEncoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig, SamplingConfig
from test_common.graph_utils import is_h100_h200
from test_common.mocks import (
    mock_estimate_memory_footprint,
    mock_pipeline_config_hf_dependencies,
    mock_pipeline_config_resolve,
)
from test_common.pipeline_model_dummy import DUMMY_LLAMA_ARCH
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
        config = PipelineConfig(model_path="test/model")

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
        config = PipelineConfig(model_path="test/model")

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
        config = PipelineConfig(model_path="test/model")

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
        config = PipelineConfig(model_path="test/model")

        kwargs = {
            "enable_lora": True,
            "lora_paths": ["/path/to/lora1", "/path/to/lora2"],
            "max_lora_rank": 32,
            "other_param": "value",
        }

        config._create_lora_config_if_needed(kwargs)

        # Should create LoRA config
        assert config.lora_config is not None
        assert config.lora_config.lora_paths == [
            "/path/to/lora1",
            "/path/to/lora2",
        ]
        assert config.lora_config.max_lora_rank == 32

        # Should remove LoRA-related kwargs
        assert "lora_paths" not in kwargs
        assert "max_lora_rank" not in kwargs
        assert "other_param" in kwargs  # Non-LoRA params should remain

    @mock_pipeline_config_resolve
    def test_create_lora_config_if_needed_error_on_incomplete_config(
        self,
    ) -> None:
        """Test error when LoRA config detected but no lora_paths provided."""
        config = PipelineConfig(model_path="test/model")

        kwargs = {
            "max_lora_rank": 32,
            "max_num_loras": 10,
        }
        config._create_lora_config_if_needed(kwargs)
        # LoRA config should not be created if no lora_paths are provided.
        assert config.lora_config is None

    @mock_pipeline_config_resolve
    def test_create_draft_model_config_if_needed_with_model_path(self) -> None:
        """Test draft model config creation when model_path is provided."""
        config = PipelineConfig(model_path="test/model")

        kwargs = {
            "draft_model_path": "/path/to/draft",
            "draft_quantization_encoding": "float32",
            "other_param": "value",
        }

        config._create_draft_model_config_if_needed(kwargs)

        # Should create draft model config
        assert config.draft_model_config is not None
        assert config.draft_model_config.model_path == "/path/to/draft"
        assert config.draft_model_config.quantization_encoding == "float32"

        # Should remove draft-related kwargs
        assert "draft_model_path" not in kwargs
        assert "draft_quantization_encoding" not in kwargs
        assert "other_param" in kwargs  # Non-draft params should remain

    @mock_pipeline_config_resolve
    def test_create_draft_model_config_if_needed_error_on_incomplete_config(
        self,
    ) -> None:
        """Test error when draft model config detected but no model_path provided."""
        config = PipelineConfig(model_path="test/model")

        kwargs = {
            "draft_quantization_encoding": "float32",
            "draft_max_length": 1024,
        }

        config._create_draft_model_config_if_needed(kwargs)
        # Draft model config should not be created if no model_path is provided.
        assert config.draft_model_config is None

    @mock_pipeline_config_resolve
    def test_create_and_set_config_basic(self) -> None:
        """Test basic config creation and setting."""
        config = PipelineConfig(model_path="test/model")

        matched_kwargs: dict[str, Any] = {
            "enable_structured_output": True,
            "enable_penalties": True,
        }
        kv_cache_kwargs: dict[str, Any] = {}

        config._create_and_set_config(
            "_sampling", SamplingConfig, matched_kwargs, kv_cache_kwargs
        )

        # Should create and set the config
        assert config.sampling_config is not None
        assert config.sampling_config.enable_structured_output is True
        assert config.sampling_config.enable_penalties is True

    @mock_pipeline_config_resolve
    def test_create_and_set_config_model_config_with_kv_cache(self) -> None:
        """Test model config creation with KV cache kwargs."""
        config = PipelineConfig(model_path="test/model")

        matched_kwargs: dict[str, Any] = {"model_path": "/test/path"}
        kv_cache_kwargs: dict[str, Any] = {"kv_cache_page_size": 256}

        config._create_and_set_config(
            "_model", MAXModelConfig, matched_kwargs, kv_cache_kwargs
        )

        # Should create model config with KV cache config
        assert config.model is not None
        assert config.model.model_path == "/test/path"
        assert config.model._kv_cache.kv_cache_page_size == 256

    @mock_pipeline_config_resolve
    def test_create_and_set_config_sampling_with_echo_enabled(self) -> None:
        """Test sampling config creation with echo enabled sets variable logits."""
        config = PipelineConfig(model_path="test/model", enable_echo=True)

        matched_kwargs = {"enable_min_tokens": True}
        kv_cache_kwargs: dict[str, Any] = {}

        config._create_and_set_config(
            "_sampling", SamplingConfig, matched_kwargs, kv_cache_kwargs
        )

        # Should create sampling config with variable logits enabled
        assert config.sampling_config is not None
        assert config.sampling_config.enable_min_tokens is True
        assert config.sampling_config.enable_variable_logits is True

    @mock_pipeline_config_resolve
    def test_process_remaining_config_classes(self) -> None:
        """Test processing of remaining config classes."""
        config = PipelineConfig(model_path="test/model")

        unmatched_kwargs = {
            "enable_structured_output": True,  # SamplingConfig
            "enable_penalties": True,  # SamplingConfig
            "model_path": "/override/path",  # MAXModelConfig
            "kv_cache_page_size": 128,  # KVCacheConfig
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
        assert config.sampling_config.enable_structured_output is True
        assert config.sampling_config.enable_penalties is True
        assert config.model.model_path == "/override/path"
        assert config.model._kv_cache.kv_cache_page_size == 128

    @mock_pipeline_config_resolve
    def test_process_remaining_config_classes_no_matches(self) -> None:
        """Test processing when no config classes match."""
        config = PipelineConfig(model_path="test/model")

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

        config = PipelineConfig(**kwargs)

        # Should have created all configs correctly
        assert config.max_batch_size == 4

        # LoRA config
        assert config.lora_config is not None
        assert config.lora_config.lora_paths == ["/lora1", "/lora2"]
        assert config.lora_config.max_lora_rank == 64

        # Draft model config
        assert config.draft_model_config is not None
        assert config.draft_model_config.model_path == "/draft/model"
        assert config.draft_model_config.quantization_encoding == "float32"

        # Sampling config
        assert config.sampling_config.enable_structured_output is True
        assert config.sampling_config.enable_penalties is False

        # Model config with KV cache
        assert config.model.quantization_encoding == "bfloat16"
        assert config.model.kv_cache_config.kv_cache_page_size == 512


@prepare_registry
@mock_estimate_memory_footprint
def test_validate_model_path__bad_repo_provided() -> None:
    # This test requires a HF call to check that this repo is not valid.
    with pytest.raises(Exception):
        _ = PipelineConfig(
            model_path="bert-base-asdfasdf",
        )


@mock_pipeline_config_hf_dependencies
def test_config_init__raises_with_no_model_path() -> None:
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(weight_path="file.gguf")


@mock_pipeline_config_hf_dependencies
def test_config_post_init__with_weight_path_but_no_model_path() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        weight_path=[
            Path(
                "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
            )
        ],
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
        model_path=llama_3_1_8b_instruct_local_path,
        weight_path=[
            Path(
                "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
            )
        ],
    )

    assert (
        config.model._weights_repo_id == "modularai/Llama-3.1-8B-Instruct-GGUF"
    )
    assert config.model.weight_path == [Path("llama-3.1-8b-instruct-q4_0.gguf")]


@mock_pipeline_config_hf_dependencies
def test_config_init__reformats_with_str_weights_path(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    # We expect this to convert the string.
    config = PipelineConfig(
        model_path=modular_ai_llama_3_1_local_path,
        weight_path=[
            Path(
                "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
            )
        ],
    )

    assert isinstance(config.model.weight_path, list)
    assert len(config.model.weight_path) == 1
    assert isinstance(config.model.weight_path[0], Path)


@pytest.mark.skipif(not is_h100_h200(), reason="This fails on MI300")
@mock_pipeline_config_hf_dependencies
def test_validate_model_path__correct_repo_id_provided(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        model_path=modular_ai_llama_3_1_local_path,
        quantization_encoding=SupportedEncoding.bfloat16,
    )

    assert config.model.model_path == modular_ai_llama_3_1_local_path


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_incompatible_quantization_encoding(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    with pytest.raises(ValueError):
        # This should raise, as q4_k != f32.
        config = PipelineConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding=SupportedEncoding.q4_k,
            weight_path=[
                Path(
                    "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                )
            ],
            max_batch_size=1,
            max_length=1,
        )

    # This should not raise, as float32 == f32.
    config = PipelineConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        quantization_encoding=SupportedEncoding.float32,
        weight_path=[
            Path(
                "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
            )
        ],
        max_batch_size=1,
        max_length=1,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
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
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding=SupportedEncoding.float32,
            max_batch_size=1,
            max_length=1,
        )

    # This should pass, because the flag also supports casting bfloat16 weights
    # to float32.
    config = PipelineConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        quantization_encoding=SupportedEncoding.float32,
        max_batch_size=1,
        max_length=1,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )

    # This should not raise, as allow_safetensors_weights_fp32_bf6_bidirectional_cast is set to True,
    # and the quantization encoding is set to bfloat16.
    config = PipelineConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        quantization_encoding=SupportedEncoding.bfloat16,
        max_batch_size=1,
        max_length=1,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )

    # Test that quantization_encoding is required when allow_safetensors_weights_fp32_bf6_bidirectional_cast is True.
    with pytest.raises(
        ValueError,
        match="--quantization-encoding must be provided when --allow-safetensors-weights-fp32-bf6-bidirectional-cast is enabled",
    ):
        config = PipelineConfig(
            model_path="test/model",
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
            # Note: quantization_encoding is not provided, which should cause the error
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
        model_path=modular_ai_llama_3_1_local_path,
        quantization_encoding=SupportedEncoding.bfloat16,
        max_batch_size=1,
        max_length=1,
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
            model_path=gemma_3_1b_it_local_path,
            max_batch_size=1,
            max_length=1,
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


@pytest.mark.skipif(not is_h100_h200(), reason="This fails on MI300")
@mock_pipeline_config_hf_dependencies
def test_config_is_picklable(
    tmp_path: Path, modular_ai_llama_3_1_local_path: str
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        model_path=modular_ai_llama_3_1_local_path,
        quantization_encoding=SupportedEncoding.bfloat16,
    )

    config.model._huggingface_config = None

    pickle_path = tmp_path / "config.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(config, f)

    with open(pickle_path, "rb") as f:
        limited_pickler = LimitedPickler(f)
        loaded_config = limited_pickler.load()

    assert loaded_config == config


@pytest.mark.skipif(not is_h100_h200(), reason="This fails on MI300")
@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_config__validates_supported_device(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Valid device/encoding combinations.
    _ = PipelineConfig(
        model_path=modular_ai_llama_3_1_local_path,
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        max_length=1,
    )

    if accelerator_count() == 0:
        with pytest.raises(ValueError):
            _ = PipelineConfig(
                model_path=modular_ai_llama_3_1_local_path,
                device_specs=[DeviceSpec.accelerator()],
                quantization_encoding=SupportedEncoding.float32,
                max_length=1,
            )
    else:
        _ = PipelineConfig(
            model_path=modular_ai_llama_3_1_local_path,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding=SupportedEncoding.bfloat16,
            max_length=1,
        )

    with pytest.raises(
        ValueError, match="not compatible with the selected device type 'cpu'"
    ):
        # Invalid device/encoding combinations.
        config = PipelineConfig(
            model_path=modular_ai_llama_3_1_local_path,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.bfloat16,
            max_length=1,
        )


@prepare_registry
def test_config__validates_lora_configuration(
    llama_3_1_8b_instruct_local_path: str, llama_3_1_8b_lora_local_path: str
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    # Test LoRA configuration with valid config
    config = PipelineConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        device_specs=[DeviceSpec.accelerator()],
        max_length=1,
        enable_lora=True,
        lora_paths=[llama_3_1_8b_lora_local_path],
        quantization_encoding=SupportedEncoding.bfloat16,
        enable_prefix_caching=False,  # Must be disabled for LoRA
    )
    assert config.lora_config is not None
    assert config.lora_config.lora_paths[0] == llama_3_1_8b_lora_local_path
    assert config.lora_config.max_lora_rank == 16
    assert config.lora_config.max_num_loras == 1


@prepare_registry
@mock_estimate_memory_footprint
def test_config__validates_lora_only_supported_for_llama(
    gemma_3_1b_it_local_path: str,
) -> None:
    """Test that LoRA validation fails for non-Llama models."""
    # Import and register Gemma architecture for testing
    from test_common.pipeline_model_dummy import DUMMY_GEMMA_ARCH

    PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH, allow_override=True)

    # Test that enabling LoRA on a non-Llama model raises ValueError
    with pytest.raises(
        ValueError,
        match=r"LoRA is not currently supported for architecture.*LoRA support is currently only available for Llama-3\.x models",
    ):
        _ = PipelineConfig(
            model_path=gemma_3_1b_it_local_path,
            device_specs=[DeviceSpec.accelerator()],
            max_length=1,
            enable_lora=True,
            lora_paths=["/some/lora/path"],
            enable_prefix_caching=False,
            quantization_encoding=SupportedEncoding.bfloat16,
        )


@prepare_registry
@mock_estimate_memory_footprint
def test_config__validates_lora_works_for_llama(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that LoRA validation passes for Llama models."""
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    config = PipelineConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        device_specs=[DeviceSpec.accelerator()],
        max_length=1,
        enable_lora=True,
        lora_paths=["/some/lora/path"],
        quantization_encoding=SupportedEncoding.bfloat16,
        enable_prefix_caching=False,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )

    # Verify LoRA config was created successfully
    assert config.lora_config is not None
    assert config.lora_config.enable_lora is True
    assert config.lora_config.lora_paths == ["/some/lora/path"]


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
            model_path=llama_3_1_8b_instruct_local_path,
            device_specs=[DeviceSpec.accelerator()],
            max_length=1,
            enable_lora=True,
            lora_paths=["/some/lora/path"],
            quantization_encoding=SupportedEncoding.bfloat16,
            enable_prefix_caching=True,  # This should conflict with LoRA
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
        model_path=llama_3_1_8b_instruct_local_path,
        enable_lora=True,
        lora_paths=["/some/lora/path"],
        enable_prefix_caching=False,  # LoRA is not compatible with prefix caching
        device_specs=[DeviceSpec.accelerator()],
        max_length=1,
        quantization_encoding=SupportedEncoding.bfloat16,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )
    assert config.lora_config is not None
    assert config.lora_config.enable_lora is True


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
            model_path=llama_3_1_8b_instruct_local_path,
            enable_lora=True,
            lora_paths=["/some/lora/path"],
            enable_prefix_caching=False,  # LoRA is not compatible with prefix caching
            device_specs=[
                DeviceSpec.accelerator(),
                DeviceSpec.accelerator(),
            ],
            max_length=1,
            quantization_encoding=SupportedEncoding.bfloat16,
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        )

    config = PipelineConfig(
        model_path=llama_3_1_8b_instruct_local_path,
        device_specs=[DeviceSpec.accelerator(), DeviceSpec.accelerator()],
        max_length=1,
        quantization_encoding=SupportedEncoding.bfloat16,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )
    assert config.lora_config is None


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
