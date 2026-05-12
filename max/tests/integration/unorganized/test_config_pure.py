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
from unittest.mock import Mock, patch

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
    PipelineRuntimeConfig,
    SamplingConfig,
)
from max.pipelines.lib.config import AudioGenerationConfig
from max.pipelines.lib.config.config_enums import SupportedEncoding
from max.pipelines.lib.config.speculative_config import SpeculativeConfig
from max.pipelines.lib.model_manifest import ModelManifest
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
        PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

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
        PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

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
        PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

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
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

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
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

        kwargs = {
            "max_lora_rank": 32,
            "max_num_loras": 10,
        }
        config._create_lora_config_if_needed(kwargs)
        # LoRA config should not be created if no lora_paths are provided.
        assert config.lora is None

    @mock_pipeline_config_resolve
    def test_create_and_set_config_basic(self) -> None:
        """Test basic config creation and setting."""
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

        matched_kwargs: dict[str, Any] = {
            "enable_structured_output": True,
            "enable_penalties": True,
        }

        config._create_and_set_config(
            "sampling", SamplingConfig, matched_kwargs
        )

        # Should create and set the config
        assert config.sampling is not None
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is True

    @mock_pipeline_config_resolve
    def test_create_and_set_config_sampling_with_echo_enabled(self) -> None:
        """Test sampling config creation with echo enabled sets variable logits."""
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="test/model", enable_echo=True
                    )
                }
            ),
        )

        matched_kwargs = {"enable_min_tokens": True}

        config._create_and_set_config(
            "sampling", SamplingConfig, matched_kwargs
        )

        # Should create sampling config with variable logits enabled
        assert config.sampling is not None
        assert config.sampling.enable_min_tokens is True
        assert config.sampling.enable_variable_logits is True

    @mock_pipeline_config_resolve
    def test_process_remaining_config_classes(self) -> None:
        """Test processing of remaining config classes."""
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

        unmatched_kwargs = {
            "enable_structured_output": True,  # SamplingConfig
            "enable_penalties": True,  # SamplingConfig
            "unknown_param": "value",  # Should remain unmatched
        }

        config._process_remaining_config_classes(unmatched_kwargs)

        # Should process and remove matched kwargs
        assert "enable_structured_output" not in unmatched_kwargs
        assert "enable_penalties" not in unmatched_kwargs

        # Should leave unmatched kwargs
        assert "unknown_param" in unmatched_kwargs

        # Should update configs
        assert config.sampling.enable_structured_output is True
        assert config.sampling.enable_penalties is True

    @mock_pipeline_config_resolve
    def test_process_remaining_config_classes_no_matches(self) -> None:
        """Test processing when no config classes match."""
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="test/model")}
            ),
        )

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
        assert config.runtime.max_batch_size == 4

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

    @mock_pipeline_config_resolve
    def test_denoising_cache_survives_runtime_kwargs(self) -> None:
        """``--taylorseer`` and friends must reach ``runtime.denoising_cache``
        even when runtime kwargs are also present.

        The CLI ``serve`` flow flattens every flag into ``PipelineConfig``
        kwargs, so taylorseer/FBC fields and runtime fields like
        ``max_batch_size`` arrive together. Cache fields must not be wiped
        when the runtime config gets reconstructed from the runtime kwargs.
        """
        kwargs = {
            "model_path": "test/model",
            # DenoisingCacheConfig fields (--taylorseer etc.)
            "taylorseer": True,
            "taylorseer_cache_interval": 5,
            "taylorseer_warmup_steps": 4,
            "taylorseer_max_order": 1,
            # PipelineRuntimeConfig field that triggers runtime reconstruction
            "max_batch_size": 4,
        }

        config = PipelineConfig(**kwargs)  # type: ignore[arg-type]

        assert config.runtime.max_batch_size == 4
        assert config.runtime.denoising_cache.taylorseer is True
        assert config.runtime.denoising_cache.taylorseer_cache_interval == 5
        assert config.runtime.denoising_cache.taylorseer_warmup_steps == 4
        assert config.runtime.denoising_cache.taylorseer_max_order == 1

    @mock_pipeline_config_resolve
    def test_first_block_caching_survives_runtime_kwargs(self) -> None:
        """``--first-block-caching`` must also survive runtime reconstruction."""
        kwargs = {
            "model_path": "test/model",
            "first_block_caching": True,
            "max_batch_size": 4,
        }

        config = PipelineConfig(**kwargs)  # type: ignore[arg-type]

        assert config.runtime.max_batch_size == 4
        assert config.runtime.denoising_cache.first_block_caching is True


class TestDraftModelDefaultsInheritance:
    """Tests that draft model inherits certain defaults from the target model."""

    def test_apply_draft_model_defaults_inherits_trust_remote_code(
        self,
    ) -> None:
        """_apply_draft_model_defaults inherits trust_remote_code from target."""
        target_model = MAXModelConfig(
            model_path="test/model",
            trust_remote_code=True,
        )
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        assert draft_kwargs["trust_remote_code"] is True

    def test_apply_draft_model_defaults_does_not_inherit_false_trust_remote_code(
        self,
    ) -> None:
        """_apply_draft_model_defaults does not inherit trust_remote_code=False."""
        target_model = MAXModelConfig(
            model_path="test/model",
            trust_remote_code=False,
        )
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        # trust_remote_code should not be added when target has False
        assert "trust_remote_code" not in draft_kwargs

    def test_apply_draft_model_defaults_preserves_explicit_trust_remote_code(
        self,
    ) -> None:
        """Explicit draft trust_remote_code is not overridden."""
        target_model = MAXModelConfig(
            model_path="test/model",
            trust_remote_code=True,
        )
        draft_kwargs: dict[str, Any] = {
            "model_path": "test/draft",
            "trust_remote_code": False,
        }

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        # Explicit False should be preserved
        assert draft_kwargs["trust_remote_code"] is False

    def test_apply_draft_model_defaults_inherits_device_specs(self) -> None:
        """_apply_draft_model_defaults inherits device_specs from target."""
        target_devices = [DeviceSpec.cpu()]
        target_model = MAXModelConfig(
            model_path="test/model",
            device_specs=target_devices,
        )
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        assert draft_kwargs["device_specs"] == target_devices

    def test_apply_draft_model_defaults_preserves_explicit_device_specs(
        self,
    ) -> None:
        """Explicit draft device_specs is not overridden."""
        target_devices = [DeviceSpec.cpu()]
        draft_devices = [DeviceSpec.accelerator()]
        target_model = MAXModelConfig(
            model_path="test/model",
            device_specs=target_devices,
        )
        draft_kwargs: dict[str, Any] = {
            "model_path": "test/draft",
            "device_specs": draft_devices,
        }

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        assert draft_kwargs["device_specs"] == draft_devices

    def test_apply_draft_model_defaults_inherits_data_parallel_degree(
        self,
    ) -> None:
        """_apply_draft_model_defaults inherits data_parallel_degree from target."""
        target_model = MAXModelConfig(
            model_path="test/model",
            data_parallel_degree=8,
        )
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        assert draft_kwargs["data_parallel_degree"] == 8

    def test_apply_draft_model_defaults_preserves_explicit_data_parallel_degree(
        self,
    ) -> None:
        """Explicit draft data_parallel_degree is not overridden."""
        target_model = MAXModelConfig(
            model_path="test/model",
            data_parallel_degree=8,
        )
        draft_kwargs: dict[str, Any] = {
            "model_path": "test/draft",
            "data_parallel_degree": 4,
        }

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        assert draft_kwargs["data_parallel_degree"] == 4

    def test_apply_draft_model_defaults_does_not_inherit_quantization_encoding(
        self,
    ) -> None:
        """_apply_draft_model_defaults does NOT inherit quantization_encoding.

        EAGLE3 and other draft models typically use bfloat16 regardless of
        the target model's quantization. The draft model should auto-detect
        its encoding from its weights, not inherit from target.
        """
        target_model = MAXModelConfig(
            model_path="test/model",
            quantization_encoding="float4_e2m1fnx2",
        )
        draft_kwargs: dict[str, Any] = {"model_path": "test/draft"}

        PipelineConfig._apply_draft_model_defaults(draft_kwargs, target_model)

        # quantization_encoding should NOT be inherited
        assert "quantization_encoding" not in draft_kwargs


class TestDraftModelQuantizationEncoding:
    """Tests that draft model quantization_encoding is independent from target."""

    _MODEL = "trl-internal-testing/tiny-random-LlamaForCausalLM"

    @staticmethod
    def _run_speculative_memory_resolution(
        config: PipelineConfig,
        *,
        draft_max_seq_len: int = 131072,
        draft_encoding: SupportedEncoding = "bfloat16",
    ) -> None:
        """Run _validate_and_resolve_speculative_memory with mocked internals.

        Mocks architecture resolution so that calling it on the target model
        sets its encoding to ``"bfloat16"`` and on the draft model sets its
        encoding to ``draft_encoding`` (simulating auto-detection from weights).

        Args:
            config: The pipeline config to resolve.
            draft_max_seq_len: Value returned by the draft arch config's
                ``get_max_seq_len()``.  Defaults to a large value so the
                clamping path is *not* exercised unless explicitly requested.
            draft_encoding: Encoding to set on the draft model during
                architecture resolution (simulating auto-detection).
        """
        mock_draft_arch_config = Mock()
        mock_draft_arch_config.get_max_seq_len.return_value = draft_max_seq_len

        mock_arch = Mock()
        mock_arch.pipeline_model.estimate_weights_size.return_value = 0
        mock_arch.config.initialize.return_value = mock_draft_arch_config

        def fake_resolve_arch(model_config: MAXModelConfig) -> Mock:
            if model_config is config.model:
                model_config.quantization_encoding = "float8_e4m3fn"
            elif model_config is config.draft_model:
                # Draft model auto-detects its own encoding from weights
                if model_config.quantization_encoding is None:
                    model_config.quantization_encoding = draft_encoding
            return mock_arch

        with (
            patch.object(
                PipelineConfig,
                "_validate_and_resolve_architecture",
                side_effect=fake_resolve_arch,
            ),
            patch.object(
                PipelineConfig,
                "_validate_and_resolve_remaining_pipeline_config",
            ),
        ):
            config._validate_and_resolve_speculative_memory()

    @mock_pipeline_config_resolve
    def test_draft_encoding_auto_detected_independently(self) -> None:
        """Draft encoding is auto-detected from weights, not inherited from target."""
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(model_path=self._MODEL),
                    "draft": MAXModelConfig(model_path=self._MODEL),
                }
            ),
            speculative=SpeculativeConfig(speculative_method="standalone"),
        )
        assert config.draft_model is not None
        assert config.draft_model.quantization_encoding is None

        # Simulate resolution where target gets float8 and draft gets bfloat16
        self._run_speculative_memory_resolution(
            config, draft_encoding="bfloat16"
        )

        # Draft should have its own encoding (bfloat16), not target's (float8)
        assert config.model.quantization_encoding == "float8_e4m3fn"
        assert config.draft_model.quantization_encoding == "bfloat16"

    @mock_pipeline_config_resolve
    def test_explicit_draft_encoding_is_preserved(self) -> None:
        """Explicit draft encoding is not overridden during resolution."""
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(model_path=self._MODEL),
                    "draft": MAXModelConfig(
                        model_path=self._MODEL,
                        quantization_encoding="float32",
                    ),
                }
            ),
            speculative=SpeculativeConfig(speculative_method="standalone"),
        )

        self._run_speculative_memory_resolution(config)

        assert config.draft_model is not None
        assert config.draft_model.quantization_encoding == "float32"

    @mock_pipeline_config_resolve
    def test_max_length_clamped_to_draft_max_seq_len(self) -> None:
        """max_length is clamped to draft arch_config.get_max_seq_len()."""
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=self._MODEL, max_length=131072
                    ),
                    "draft": MAXModelConfig(model_path=self._MODEL),
                }
            ),
            speculative=SpeculativeConfig(speculative_method="standalone"),
        )

        self._run_speculative_memory_resolution(config, draft_max_seq_len=2048)

        assert config.model.max_length == 2048
        assert config.draft_model is not None
        assert config.draft_model.max_length == 2048


@prepare_registry
@mock_estimate_memory_footprint
def test_validate_model_path__bad_repo_provided() -> None:
    # This test requires a HF call to check that this repo is not valid.
    with pytest.raises(Exception):
        _ = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path="bert-base-asdfasdf")}
            ),
        )


def test_config_init__raises_with_no_model_path() -> None:
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(weight_path=[Path("file.gguf")])}
            ),
        )


@prepare_registry
def test_config_post_init__with_weight_path_but_no_model_path() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                        )
                    ],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                        )
                    ],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-q4_0.gguf"
                        )
                    ],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    assert config.model.model_path == modular_ai_llama_3_1_local_path


@mock_estimate_memory_footprint
def test_config__test_incompatible_quantization_encoding(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH, allow_override=True)

    with pytest.raises(ValueError):
        # This should raise, as q4_k != f32.
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        quantization_encoding="q4_k",
                        weight_path=[
                            Path(
                                "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                            )
                        ],
                        max_length=1,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                max_batch_size=1,
                prefer_module_v3=True,
            ),
        )

    # This should not raise, as float32 == f32.
    PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="float32",
                    weight_path=[
                        Path(
                            "modularai/Llama-3.1-8B-Instruct-GGUF/llama-3.1-8b-instruct-f32.gguf"
                        )
                    ],
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
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

    # Float32 <-> bfloat16 casting is always enabled,
    # so this should succeed by casting bfloat16 weights to float32.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="float32",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )
    assert config.model.kv_cache.cache_dtype == DType.float32


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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="float32",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
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

    # This should not raise, as float32 <-> bfloat16 casting is always enabled
    # and the quantization encoding is set to bfloat16.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    quantization_encoding="bfloat16",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
    )
    assert config.model.kv_cache.cache_dtype == DType.bfloat16


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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=1,
            prefer_module_v3=True,
        ),
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
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=gemma_3_1b_it_local_path, max_length=1
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                max_batch_size=1,
                prefer_module_v3=True,
            ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    device_specs=[DeviceSpec.cpu()],
                    quantization_encoding="float32",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    if accelerator_count() == 0:
        with pytest.raises(ValueError):
            _ = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path=modular_ai_llama_3_1_local_path,
                            device_specs=[DeviceSpec.accelerator()],
                            quantization_encoding="float32",
                            max_length=1,
                        )
                    }
                ),
                runtime=PipelineRuntimeConfig(
                    prefer_module_v3=True,
                ),
            )
    else:
        _ = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=modular_ai_llama_3_1_local_path,
                        device_specs=[DeviceSpec.accelerator()],
                        quantization_encoding="bfloat16",
                        max_length=1,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )

    with pytest.raises(
        ValueError, match="not compatible with the selected device type 'cpu'"
    ):
        # Invalid device/encoding combinations.
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=modular_ai_llama_3_1_local_path,
                        device_specs=[DeviceSpec.cpu()],
                        quantization_encoding="bfloat16",
                        max_length=1,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="bfloat16",
                    kv_cache=KVCacheConfig(enable_prefix_caching=False),
                    max_length=1,
                )
            }
        ),
        lora=LoRAConfig(
            enable_lora=True, lora_paths=[llama_3_1_8b_lora_local_path]
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=gemma_3_1b_it_local_path,
                        device_specs=[DeviceSpec.accelerator()],
                        quantization_encoding="bfloat16",
                        kv_cache=KVCacheConfig(enable_prefix_caching=False),
                        max_length=1,
                    )
                }
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="bfloat16",
                    kv_cache=KVCacheConfig(enable_prefix_caching=False),
                    max_length=1,
                )
            }
        ),
        lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        device_specs=[DeviceSpec.accelerator()],
                        quantization_encoding="bfloat16",
                        kv_cache=KVCacheConfig(enable_prefix_caching=True),
                        max_length=1,
                    )
                }
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
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
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[DeviceSpec.accelerator()],
                    quantization_encoding="bfloat16",
                    kv_cache=KVCacheConfig(enable_prefix_caching=False),
                    max_length=1,
                )
            }
        ),
        lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
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
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        device_specs=[
                            DeviceSpec.accelerator(),
                            DeviceSpec.accelerator(),
                        ],
                        quantization_encoding="bfloat16",
                        kv_cache=KVCacheConfig(enable_prefix_caching=False),
                        max_length=1,
                    )
                }
            ),
            lora=LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"]),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=llama_3_1_8b_instruct_local_path,
                    device_specs=[
                        DeviceSpec.accelerator(),
                        DeviceSpec.accelerator(),
                    ],
                    quantization_encoding="bfloat16",
                    max_length=1,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )
    assert config.lora is None


def test_manifest_discovers_diffusion_components() -> None:
    """Test that ModelManifest discovers components for a diffusion pipeline."""
    from transformers import PretrainedConfig

    diffusion_model = "hf-internal-testing/tiny-stable-diffusion-torch"

    manifest = ModelManifest.from_model_path(diffusion_model)

    # ModelManifest should have discovered per-component configs.
    expected_components = ["vae", "unet", "text_encoder"]
    for component in expected_components:
        assert component in manifest, (
            f"manifest should contain {component} component"
        )
        # Each component should have a valid huggingface_config.
        assert isinstance(
            manifest[component].huggingface_config, PretrainedConfig
        )

    # Metadata should contain the pipeline class name.
    assert "_class_name" in manifest.metadata
    assert "StableDiffusion" in manifest.metadata["_class_name"]


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
@pytest.mark.parametrize(
    "arch_name,max_batch_size,force,is_cuda,expected_device_graph_capture",
    [
        ("LlamaForCausalLM", 16, False, True, True),
        ("DeepseekV2ForCausalLM", 16, False, True, True),
        ("DeepseekV3ForCausalLM", 16, False, True, True),
        ("DeepseekV32ForCausalLM", 16, False, True, True),
        ("DeepseekV3ForCausalLMNextN", 16, False, True, True),
        ("KimiK25ForConditionalGeneration", 16, False, True, True),
        ("UnifiedEagleLlama3ForCausalLM", 16, False, True, True),
        ("LlamaForCausalLM", 16, False, False, False),
        ("LlamaForCausalLM", None, False, True, False),
        ("LlamaForCausalLM", 16, True, True, False),
        ("SomeOtherArchitecture", 16, False, True, False),
    ],
)
def test_validate_and_resolve_overlap_scheduler__auto_enable_device_graph_capture(
    arch_name: str,
    max_batch_size: int | None,
    force: bool,
    is_cuda: bool,
    expected_device_graph_capture: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock architecture_name so we don't reach out to HF for the config.
    monkeypatch.setattr(
        MAXModelConfig, "architecture_name", property(lambda self: arch_name)
    )
    # Force PIPELINE_REGISTRY.retrieve_architecture to return a custom arch.
    arch = SimpleNamespace(name=arch_name)
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch),
    )
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda" if is_cuda else "hip"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_num_steps=42,
            force=force,
            max_batch_size=max_batch_size,
        ),
    )
    config._validate_and_resolve_overlap_scheduler()

    assert config.runtime.device_graph_capture is expected_device_graph_capture
    if expected_device_graph_capture:
        assert config.runtime.enable_overlap_scheduler is True
        assert config.runtime.max_num_steps == 1


@prepare_registry
@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__no_device_graph_capture_for_prefill_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device graph capture is not supported for prefill-only workers."""
    arch_name = "LlamaForCausalLM"
    monkeypatch.setattr(
        MAXModelConfig, "architecture_name", property(lambda self: arch_name)
    )
    arch = SimpleNamespace(name=arch_name)
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch),
    )
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_num_steps=42,
            max_batch_size=16,
            pipeline_role="prefill_only",
        ),
    )
    config._validate_and_resolve_overlap_scheduler()

    # Overlap scheduling should be auto-enabled for prefill_only.
    assert config.runtime.enable_overlap_scheduler is True
    assert config.runtime.max_num_steps == 1
    # But device graph capture should NOT be auto-enabled.
    assert config.runtime.device_graph_capture is False


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
            # Mock architecture_name so we don't reach out to HF for the config.
            m.setattr(
                MAXModelConfig,
                "architecture_name",
                property(lambda self: arch_name),
            )
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
        "LlamaForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
    ):
        with patch_retrieve_architecture(arch_name):
            config = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path="test/model",
                            device_specs=[DeviceSpec.accelerator()],
                        )
                    }
                ),
                runtime=PipelineRuntimeConfig(max_num_steps=42),
            )
            config._validate_and_resolve_overlap_scheduler()
            assert config.runtime.enable_overlap_scheduler is True
            assert config.runtime.max_num_steps == 1

    # Don't override if the device is CPU
    with patch_retrieve_architecture("LlamaForCausalLM"):
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="test/model",
                        device_specs=[DeviceSpec.cpu()],
                    )
                }
            ),
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.runtime.enable_overlap_scheduler is False

    # Don't override if structured output is enabled
    with patch_retrieve_architecture("LlamaForCausalLM"):
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="test/model",
                        device_specs=[DeviceSpec.accelerator()],
                    )
                }
            ),
            sampling=SamplingConfig(enable_structured_output=True),
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.runtime.enable_overlap_scheduler is False

    # Auto-enable for DI pipeline roles (prefill_only, decode_only)
    for role in ("prefill_only", "decode_only"):
        with patch_retrieve_architecture("LlamaForCausalLM"):
            config = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path="test/model",
                            device_specs=[DeviceSpec.accelerator()],
                        )
                    }
                ),
                runtime=PipelineRuntimeConfig(pipeline_role=role),
            )
            config._validate_and_resolve_overlap_scheduler()
            assert config.runtime.enable_overlap_scheduler is True
            assert config.runtime.max_num_steps == 1

    # Don't override for other architectures
    with patch_retrieve_architecture("SomeOtherArchitecture"):
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="test/model",
                        device_specs=[DeviceSpec.accelerator()],
                    )
                }
            ),
        )
        config._validate_and_resolve_overlap_scheduler()
        assert config.runtime.enable_overlap_scheduler is False


@prepare_registry
@mock_pipeline_config_resolve
def test_validate_and_resolve_overlap_scheduler__validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock architecture_name so we don't reach out to HF for the config.
    monkeypatch.setattr(
        MAXModelConfig,
        "architecture_name",
        property(lambda self: "SomeArchitecture"),
    )

    # Allow user to manually enable overlap scheduler
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(enable_overlap_scheduler=True),
    )
    config._validate_and_resolve_overlap_scheduler()
    assert config.runtime.enable_overlap_scheduler is True

    # Error out if user tries to enable overlap scheduler on CPU
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.cpu()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(enable_overlap_scheduler=True),
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # prefill_only with overlap scheduler is now allowed (experimental),
    # the runtime just logs a warning and sets max_num_steps=1.
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            pipeline_role="prefill_only",
            enable_overlap_scheduler=True,
        ),
    )
    config._validate_and_resolve_overlap_scheduler()
    assert config.runtime.enable_overlap_scheduler is True
    assert config.runtime.max_num_steps == 1

    # Error out if user tries to enable overlap scheduler with AudioGenerationConfig
    config = AudioGenerationConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            pipeline_role="prefill_and_decode",
            enable_overlap_scheduler=True,
        ),
        audio_decoder=Mock(),
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()

    # Error out if user tries to enable overlap scheduler with structured output
    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        runtime=PipelineRuntimeConfig(enable_overlap_scheduler=True),
    )
    with pytest.raises(ValueError):
        config._validate_and_resolve_overlap_scheduler()


@prepare_registry
@mock_pipeline_config_resolve
@pytest.mark.parametrize(
    "num_speculative_tokens,expected_device_graph_capture",
    [
        (1, True),
        (2, False),
        (5, False),
    ],
    ids=["1_spec_token", "2_spec_tokens", "5_spec_tokens"],
)
def test_auto_device_graph_capture_eagle_gating(
    num_speculative_tokens: int,
    expected_device_graph_capture: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eagle arch auto-enables graph capture only when num_speculative_tokens <= 1."""
    monkeypatch.setattr(MAXModelConfig, "huggingface_model_repo", Mock())
    arch = SimpleNamespace(name="UnifiedEagleLlama3ForCausalLM")
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch),
    )
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda"),
    )

    config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="test/model",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        speculative=SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=num_speculative_tokens,
        ),
        runtime=PipelineRuntimeConfig(max_batch_size=16),
    )
    config._validate_and_resolve_overlap_scheduler()

    assert config.runtime.device_graph_capture is expected_device_graph_capture


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__applies_arch_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user did not set runtime.reasoning_parser and the resolved
    architecture declares a default, the default is applied."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        reasoning_parser="kimik2_5",
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch),
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )
    config.models["main"]._huggingface_config = SimpleNamespace(
        architectures=["KimiK25ForConditionalGeneration"]
    )
    assert config.runtime.reasoning_parser is None

    config._resolve_default_reasoning_parser()

    assert config.runtime.reasoning_parser == "kimik2_5"


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__user_value_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit runtime.reasoning_parser value is never overwritten,
    even when the architecture declares a different default."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        reasoning_parser="kimik2_5",
    )
    retrieve_mock = Mock(return_value=arch)
    monkeypatch.setattr(
        PIPELINE_REGISTRY, "retrieve_architecture", retrieve_mock
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(reasoning_parser="user_choice"),
    )
    config.models["main"]._huggingface_config = SimpleNamespace(
        architectures=["KimiK25ForConditionalGeneration"]
    )

    config._resolve_default_reasoning_parser()

    assert config.runtime.reasoning_parser == "user_choice"
    retrieve_mock.assert_not_called()


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_reasoning_parser__no_arch_default_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the architecture does not declare a default reasoning parser
    (or no architecture is found), runtime.reasoning_parser stays None."""
    arch_without_default = SimpleNamespace(
        name="LlamaForCausalLM",
        reasoning_parser=None,
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch_without_default),
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )
    config.models["main"]._huggingface_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"]
    )

    config._resolve_default_reasoning_parser()
    assert config.runtime.reasoning_parser is None

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=None),
    )
    config._resolve_default_reasoning_parser()
    assert config.runtime.reasoning_parser is None


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__applies_arch_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When runtime.tool_parser is unset and architecture declares a default,
    the default is applied."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        tool_parser="kimik2_5",
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch),
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )
    config.models["main"]._huggingface_config = SimpleNamespace(
        architectures=["KimiK25ForConditionalGeneration"]
    )
    assert config.runtime.tool_parser is None

    config._resolve_default_tool_parser()

    assert config.runtime.tool_parser == "kimik2_5"


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__user_value_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit runtime.tool_parser value is never overwritten."""
    arch = SimpleNamespace(
        name="KimiK25ForConditionalGeneration",
        tool_parser="kimik2_5",
    )
    retrieve_mock = Mock(return_value=arch)
    monkeypatch.setattr(
        PIPELINE_REGISTRY, "retrieve_architecture", retrieve_mock
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(tool_parser="user_choice"),
    )
    config.models["main"]._huggingface_config = SimpleNamespace(
        architectures=["KimiK25ForConditionalGeneration"]
    )

    config._resolve_default_tool_parser()

    assert config.runtime.tool_parser == "user_choice"
    retrieve_mock.assert_not_called()


@prepare_registry
@mock_pipeline_config_resolve
def test_resolve_default_tool_parser__no_arch_default_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If architecture has no default tool parser, runtime value stays None."""
    arch_without_default = SimpleNamespace(
        name="LlamaForCausalLM",
        tool_parser=None,
    )
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=arch_without_default),
    )

    config = PipelineConfig(
        models=ModelManifest({"main": MAXModelConfig(model_path="test/model")}),
        runtime=PipelineRuntimeConfig(),
    )
    config.models["main"]._huggingface_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"]
    )

    config._resolve_default_tool_parser()
    assert config.runtime.tool_parser is None

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_architecture",
        Mock(return_value=None),
    )
    config._resolve_default_tool_parser()
    assert config.runtime.tool_parser is None
