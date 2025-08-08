# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for MAXConfig interface."""

import tempfile
from dataclasses import dataclass

import pytest
import yaml
from conftest import assert_help_covers_all_public_fields
from max.dtype import DType
from max.engine import GPUProfilingMode
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    MAXConfig,
    ProfilingConfig,
    SamplingConfig,
    convert_max_config_value,
    get_default_max_config_file_section_name,
)


@dataclass
class TestConfig(MAXConfig):
    """Test MAXConfig class for unit testing."""

    _config_file_section_name: str = "test_config"

    test_field: str = "default_value"
    test_int: int = 42
    test_bool: bool = True

    def help(self) -> dict[str, str]:
        return {
            "test_field": "A test string field",
            "test_int": "A test integer field",
            "test_bool": "A test boolean field",
        }


@dataclass
class MissingSectionNameConfig(MAXConfig):
    """MAXConfig without _config_file_section_name for testing."""

    test_field: str = "value"

    def help(self) -> dict[str, str]:
        return {"test_field": "A test field"}


class TestMAXConfigInterface:
    """Test suite for MAXConfig base interface."""

    def test_config_file_section_name_required(self) -> None:
        """Test that _config_file_section_name is required."""
        with pytest.raises(
            ValueError,
            match="must define a '_config_file_section_name' class attribute",
        ):
            get_default_max_config_file_section_name(MissingSectionNameConfig)

    def test_get_default_max_config_file_section_name(self) -> None:
        """Test get_default_max_config_file_section_name method."""
        section_name = get_default_max_config_file_section_name(TestConfig)
        assert section_name == "test_config"

    def test_abstract_help_method(self) -> None:
        """Test that help method is abstract and must be implemented."""
        # TestConfig implements help, so it should work.
        config = TestConfig()
        help_dict = config.help()
        assert isinstance(help_dict, dict)
        assert "test_field" in help_dict

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
            config_path = f.name

            config = TestConfig.from_config_file(config_path)
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
            },
            "other_section": {
                "other_field": "other_value",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config_path = f.name

            config = TestConfig.from_config_file(config_path)
            assert config.test_field == "full_value"
            assert config.test_int == 200
            assert config.test_bool is True  # Default value.

    def test_file_not_found_error(self) -> None:
        """Test FileNotFoundError when MAXConfig file doesn't exist."""
        with pytest.raises(
            FileNotFoundError, match="Configuration file not found"
        ):
            TestConfig.from_config_file("nonexistent_file.yaml")

    def test_invalid_yaml_error(self) -> None:
        """Test ValueError when YAML file is invalid."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML.
            f.flush()  # Ensure data is written to disk
            config_path = f.name

            with pytest.raises(
                ValueError, match="Failed to load configuration"
            ):
                TestConfig.from_config_file(config_path)

    def test_section_not_found_error(self) -> None:
        """Test ValueError when specified section is not found."""
        config_data = {"other_section": {"field": "value"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config_path = f.name

            with pytest.raises(
                ValueError, match="Section 'test_config' not found"
            ):
                TestConfig.from_config_file(config_path)

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
            config_path = f.name

            # Test KVCacheConfig enum loading.
            kv_config = KVCacheConfig.from_config_file(config_path)
            assert kv_config.cache_strategy == KVCacheStrategy.PAGED
            assert kv_config.kv_cache_page_size == 256
            assert kv_config.enable_kvcache_swapping_to_host is True

            # Test ProfilingConfig enum loading.
            profiling_config = ProfilingConfig.from_config_file(config_path)
            assert profiling_config.gpu_profiling == GPUProfilingMode.ON

            # Test SamplingConfig enum loading.
            sampling_config = SamplingConfig.from_config_file(config_path)
            assert isinstance(sampling_config.in_dtype, DType)
            assert sampling_config.in_dtype == DType.float16
            assert sampling_config.out_dtype == DType.bfloat16


class TestMAXConfigTypeConversion:
    """Test suite for MAXConfig type conversion functionality."""

    def test_basic_type_conversion(self) -> None:
        """Test conversion of basic types."""
        # String to int.
        assert convert_max_config_value("42", int, "test_field") == 42

        # String to bool.
        assert convert_max_config_value("true", bool, "test_field") is True
        assert convert_max_config_value("false", bool, "test_field") is False

        # String to float.
        assert convert_max_config_value("3.14", float, "test_field") == 3.14

    def test_none_value_handling(self) -> None:
        """Test handling of None values."""
        result = convert_max_config_value(None, str, "test_field")
        assert result is None

    def test_boolean_string_variations(self) -> None:
        """Test various boolean string representations."""
        # Test true values.
        true_values = [
            "true",
            "True",
            "TRUE",
            "1",
            "yes",
            "Yes",
            "YES",
            "on",
            "On",
            "ON",
        ]
        for value in true_values:
            result = convert_max_config_value(value, bool, "test_field")
            assert result is True, f"Expected True for '{value}'"

        # Test false values.
        false_values = [
            "false",
            "False",
            "FALSE",
            "0",
            "no",
            "No",
            "NO",
            "off",
            "Off",
            "OFF",
        ]
        for value in false_values:
            result = convert_max_config_value(value, bool, "test_field")
            assert result is False, f"Expected False for '{value}'"

    def test_boolean_numeric_values(self) -> None:
        """Test boolean conversion from numeric values."""
        # Test integer values.
        assert convert_max_config_value(1, bool, "test_field") is True
        assert convert_max_config_value(0, bool, "test_field") is False
        assert convert_max_config_value(42, bool, "test_field") is True

        # Test float values.
        assert convert_max_config_value(1.0, bool, "test_field") is True
        assert convert_max_config_value(0.0, bool, "test_field") is False
        assert convert_max_config_value(3.14, bool, "test_field") is True

    def test_boolean_invalid_values(self) -> None:
        """Test boolean conversion with invalid string values."""
        invalid_values = ["maybe", "unknown", "invalid", "2"]
        for value in invalid_values:
            with pytest.raises(ValueError, match="Cannot convert .* to bool"):
                convert_max_config_value(value, bool, "test_field")

    def test_enum_conversion_gpu_profiling_mode(self) -> None:
        """Test conversion to GPUProfilingMode enum."""
        # Test by name (case-insensitive).
        result = convert_max_config_value(
            "OFF", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.OFF

        result = convert_max_config_value(
            "detailed", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.DETAILED

        result = convert_max_config_value(
            "on", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.ON

        # Test case insensitive
        result = convert_max_config_value(
            "detailed", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.DETAILED

        # Test with existing enum value.
        result = convert_max_config_value(
            GPUProfilingMode.ON, GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.ON

    def test_enum_conversion_kv_cache_strategy(self) -> None:
        """Test conversion to KVCacheStrategy enum."""
        # Test by name.
        result = convert_max_config_value(
            value="PAGED",
            field_type=KVCacheStrategy,
            field_name="cache_strategy",
        )
        assert result == KVCacheStrategy.PAGED

        result = convert_max_config_value(
            value="continuous",
            field_type=KVCacheStrategy,
            field_name="cache_strategy",
        )
        assert result == KVCacheStrategy.CONTINUOUS

        result = convert_max_config_value(
            value="model_default",
            field_type=KVCacheStrategy,
            field_name="cache_strategy",
        )
        assert result == KVCacheStrategy.MODEL_DEFAULT

        # Test case insensitive
        result = convert_max_config_value(
            "CONTINUOUS", KVCacheStrategy, "cache_strategy"
        )
        assert result == KVCacheStrategy.CONTINUOUS

    def test_enum_conversion_dtype(self) -> None:
        """Test conversion to DType enum."""
        # Test common dtypes by string.
        result = convert_max_config_value("float32", DType, "dtype")
        assert result == DType.float32

        result = convert_max_config_value("float16", DType, "dtype")
        assert result == DType.float16

        result = convert_max_config_value("int8", DType, "dtype")
        assert result == DType.int8

        result = convert_max_config_value("bfloat16", DType, "dtype")
        assert result == DType.bfloat16

        # Test with existing DType value.
        result = convert_max_config_value(DType.float32, DType, "dtype")
        assert result == DType.float32

    def test_enum_conversion_invalid_values(self) -> None:
        """Test enum conversion with invalid values."""
        # Test invalid GPUProfilingMode.
        with pytest.raises(ValueError):
            convert_max_config_value(
                "invalid_mode", GPUProfilingMode, "gpu_profiling"
            )

        # Test invalid KVCacheStrategy.
        with pytest.raises(ValueError):
            convert_max_config_value(
                "invalid_strategy", KVCacheStrategy, "cache_strategy"
            )

    def test_list_type_conversion(self) -> None:
        """Test conversion of list types."""
        # Test list of strings.
        result = convert_max_config_value(
            ["path1", "path2", "path3"], list[str], "lora_paths"
        )
        assert result == ["path1", "path2", "path3"]

        # Test list of integers.
        result = convert_max_config_value(
            ["1", "2", "3"], list[int], "int_list"
        )
        assert result == [1, 2, 3]

        # Test list of booleans.
        result = convert_max_config_value(
            ["true", "false", "1", "0"], list[bool], "bool_list"
        )
        assert result == [True, False, True, False]

    def test_list_type_invalid_input(self) -> None:
        """Test list type conversion with invalid input."""
        # Test non-list input.
        with pytest.raises(ValueError, match="Expected list"):
            convert_max_config_value("not_a_list", list[str], "test_field")

    def test_optional_type_conversion(self) -> None:
        """Test conversion of Optional types."""
        from typing import Optional

        # Test Optional[int] with valid value.
        result = convert_max_config_value("42", Optional[int], "optional_int")
        assert result == 42

        # Test Optional[int] with None.
        result = convert_max_config_value(None, Optional[int], "optional_int")
        assert result is None

        # Test Optional[bool] with valid value.
        result = convert_max_config_value(
            "true", Optional[bool], "optional_bool"
        )
        assert result is True

        # Test Optional[bool] with None.
        result = convert_max_config_value(None, Optional[bool], "optional_bool")
        assert result is None

    def test_union_type_conversion(self) -> None:
        """Test conversion of Union types."""
        from typing import Union

        # Test Union[int, str] with int.
        result = convert_max_config_value("42", Union[int, str], "union_field")
        assert result == 42

        # Test Union[int, str] with string that can't be converted to int.
        result = convert_max_config_value(
            "hello", Union[int, str], "union_field"
        )
        assert result == "hello"

        # Test Union[bool, str] with boolean string.
        result = convert_max_config_value(
            "true", Union[bool, str], "union_field"
        )
        assert result is True

    def test_complex_nested_types(self) -> None:
        """Test conversion of complex nested types."""
        from typing import Optional

        # Test Optional[list[str]].
        result = convert_max_config_value(
            ["item1", "item2"], Optional[list[str]], "optional_list"
        )
        assert result == ["item1", "item2"]

        # Test Optional[list[str]] with None.
        result = convert_max_config_value(
            None, Optional[list[str]], "optional_list"
        )
        assert result is None

    def test_direct_type_instantiation_fallback(self) -> None:
        """Test fallback to direct type instantiation."""
        # Test with a simple type that should work with direct instantiation.
        result = convert_max_config_value("3.14159", float, "pi")
        assert result == 3.14159

        # Test with integer.
        result = convert_max_config_value("42", int, "answer")
        assert result == 42


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

        # Test that help method covers all public fields.
        assert_help_covers_all_public_fields(config, "KVCacheConfig")

    def test_sampling_config(self) -> None:
        """Test SamplingConfig functionality."""
        config = SamplingConfig()

        # Test default values.
        assert hasattr(config, "in_dtype")
        assert hasattr(config, "out_dtype")

        # Test section name.
        assert config._config_file_section_name == "sampling_config"

        # Test that help method covers all public fields.
        assert_help_covers_all_public_fields(config, "SamplingConfig")

    def test_profiling_config(self) -> None:
        """Test ProfilingConfig functionality."""
        config = ProfilingConfig()

        # Test default values.
        assert hasattr(config, "gpu_profiling")

        # Test section name.
        assert config._config_file_section_name == "profiling_config"

        # Test that help method covers all public fields.
        assert_help_covers_all_public_fields(config, "ProfilingConfig")

    def test_lora_config(self) -> None:
        """Test LoRAConfig functionality."""
        config = LoRAConfig()

        # Test default values.
        assert hasattr(config, "enable_lora")
        assert hasattr(config, "lora_paths")
        assert hasattr(config, "max_lora_rank")

        # Test section name.
        assert config._config_file_section_name == "lora_config"

        # Test that help method covers all public fields.
        assert_help_covers_all_public_fields(config, "LoRAConfig")
