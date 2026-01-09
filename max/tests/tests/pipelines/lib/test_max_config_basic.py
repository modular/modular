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
"""Tests for MAXConfig interface."""

from __future__ import annotations

import tempfile

import yaml
from max.config import ConfigFileModel
from max.dtype import DType
from max.engine import GPUProfilingMode
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    ProfilingConfig,
    SamplingConfig,
)
from pydantic import Field


class TestConfig(ConfigFileModel):
    """Test ConfigFileModel class for unit testing."""

    test_field: str = Field(default="default_value")
    test_int: int = Field(default=42)
    test_bool: bool = Field(default=True)
    test_inf: float = Field(default=float("inf"))


class TestMAXConfigInterface:
    """Test suite for MAXConfig base interface."""

    def test_config_instantiation_with_defaults(self) -> None:
        """Test that MAXConfig classes can be instantiated with defaults."""
        config = TestConfig()
        assert config.test_field == "default_value"
        assert config.test_int == 42
        assert config.test_bool is True


class TestMAXConfigFileLoading:
    """Test suite for YAML MAXConfig file loading functionality."""

    def test_load_individual_config_file(self) -> None:
        """Test loading from individual MAXConfig file."""
        config_data = {
            "test_field": "loaded_value",
            "test_int": 100,
            "test_bool": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config = TestConfig(config_file=f.name)
            assert config.test_field == "loaded_value"
            assert config.test_int == 100
            assert config.test_bool is False

    def test_load_full_config_file(self) -> None:
        """Test loading from full MAXConfig file with multiple sections."""
        config_data = {
            "name": "test_full_config",
            "description": "A test full config",
            "version": "1.0",
            "test_config": {
                "test_field": "full_value",
                "test_int": 200,
                "test_inf": "inf",
            },
            "other_section": {
                "other_field": "other_value",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config = TestConfig(config_file=f.name, section_name="test_config")
            assert config.test_field == "full_value"
            assert config.test_int == 200
            assert config.test_bool is True  # Default value.
            assert config.test_inf == float("inf")

    def test_load_config_with_enums(self) -> None:
        """Test loading MAXConfig file with enum values."""
        config_data = {
            "name": "test_enum_config",
            "description": "A test config with enums",
            "version": "1.0",
            "kv_cache_config": {
                "cache_strategy": "paged",
                "kv_cache_page_size": 256,
                "enable_kvcache_swapping_to_host": True,
            },
            "profiling_config": {
                "gpu_profiling": "on",
            },
            "sampling_config": {
                "in_dtype": "float16",
                "out_dtype": "bfloat16",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            # Test KVCacheConfig enum loading.
            kv_config = KVCacheConfig(
                config_file=f.name, section_name="kv_cache_config"
            )
            assert kv_config.cache_strategy == KVCacheStrategy.PAGED
            assert kv_config.kv_cache_page_size == 256
            assert kv_config.enable_kvcache_swapping_to_host is True

            # Test ProfilingConfig enum loading.
            profiling_config = ProfilingConfig(
                config_file=f.name, section_name="profiling_config"
            )
            assert profiling_config.gpu_profiling == GPUProfilingMode.ON

            # Test SamplingConfig enum loading.
            sampling_config = SamplingConfig(
                config_file=f.name, section_name="sampling_config"
            )
            assert isinstance(sampling_config.in_dtype, DType)
            assert sampling_config.in_dtype == DType.float16
            assert sampling_config.out_dtype == DType.bfloat16


class TestBuiltinConfigClasses:
    """Test suite for built-in MAXConfig classes."""

    def test_kv_cache_config(self) -> None:
        """Test KVCacheConfig functionality."""
        config = KVCacheConfig()

        # Test default values.
        assert hasattr(config, "cache_strategy")
        assert hasattr(config, "kv_cache_page_size")

        # Test section name.
        assert config._config_file_section_name == "kv_cache_config"

    def test_sampling_config(self) -> None:
        """Test SamplingConfig functionality."""
        config = SamplingConfig()

        # Test default values.
        assert hasattr(config, "in_dtype")
        assert hasattr(config, "out_dtype")

        # Test section name.
        assert config._config_file_section_name == "sampling_config"

    def test_profiling_config(self) -> None:
        """Test ProfilingConfig functionality."""
        config = ProfilingConfig()

        # Test default values.
        assert hasattr(config, "gpu_profiling")

        # Test section name.
        assert config._config_file_section_name == "profiling_config"

    def test_lora_config(self) -> None:
        """Test LoRAConfig functionality."""
        config = LoRAConfig()

        # Test default values.
        assert hasattr(config, "enable_lora")
        assert hasattr(config, "lora_paths")
        assert hasattr(config, "max_lora_rank")

        # Test section name.
        assert config._config_file_section_name == "lora_config"
