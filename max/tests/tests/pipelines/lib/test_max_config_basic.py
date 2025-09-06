# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for MAXConfig interface."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, field, fields
from typing import Optional, Union

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
    test_inf: float = float("inf")

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "test_field": "A test string field",
            "test_int": "A test integer field",
            "test_bool": "A test boolean field",
        }


@dataclass
class MissingSectionNameConfig(MAXConfig):
    """MAXConfig without _config_file_section_name for testing."""

    test_field: str = "value"

    @staticmethod
    def help() -> dict[str, str]:
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
                "test_inf": "inf",
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
            assert config.test_inf == float("inf")

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
        assert (
            convert_max_config_value(TestConfig, "42", int, "test_field") == 42
        )

        # String to bool.
        assert (
            convert_max_config_value(TestConfig, "true", bool, "test_field")
            is True
        )
        assert (
            convert_max_config_value(TestConfig, "false", bool, "test_field")
            is False
        )

        # String to float.
        assert (
            convert_max_config_value(TestConfig, "3.14", float, "test_field")
            == 3.14
        )

    def test_none_value_handling(self) -> None:
        """Test handling of None values."""
        result = convert_max_config_value(TestConfig, None, str, "test_field")
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
            result = convert_max_config_value(
                TestConfig, value, bool, "test_field"
            )
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
            result = convert_max_config_value(
                TestConfig, value, bool, "test_field"
            )
            assert result is False, f"Expected False for '{value}'"

    def test_boolean_numeric_values(self) -> None:
        """Test boolean conversion from numeric values."""
        # Test integer values.
        assert (
            convert_max_config_value(TestConfig, 1, bool, "test_field") is True
        )
        assert (
            convert_max_config_value(TestConfig, 0, bool, "test_field") is False
        )
        assert (
            convert_max_config_value(TestConfig, 42, bool, "test_field") is True
        )

        # Test float values.
        assert (
            convert_max_config_value(TestConfig, 1.0, bool, "test_field")
            is True
        )
        assert (
            convert_max_config_value(TestConfig, 0.0, bool, "test_field")
            is False
        )
        assert (
            convert_max_config_value(TestConfig, 3.14, bool, "test_field")
            is True
        )

    def test_boolean_invalid_values(self) -> None:
        """Test boolean conversion with invalid string values."""
        invalid_values = ["maybe", "unknown", "invalid", "2"]
        for value in invalid_values:
            with pytest.raises(ValueError, match="Cannot convert .* to bool"):
                convert_max_config_value(TestConfig, value, bool, "test_field")

    def test_enum_conversion_gpu_profiling_mode(self) -> None:
        """Test conversion to GPUProfilingMode enum."""
        # Test by name (case-insensitive).
        result = convert_max_config_value(
            TestConfig, "OFF", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.OFF

        result = convert_max_config_value(
            TestConfig, "detailed", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.DETAILED

        result = convert_max_config_value(
            TestConfig, "on", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.ON

        # Test case insensitive
        result = convert_max_config_value(
            TestConfig, "detailed", GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.DETAILED

        # Test with existing enum value.
        result = convert_max_config_value(
            TestConfig, GPUProfilingMode.ON, GPUProfilingMode, "gpu_profiling"
        )
        assert result == GPUProfilingMode.ON

    def test_enum_conversion_kv_cache_strategy(self) -> None:
        """Test conversion to KVCacheStrategy enum."""
        # Test by name.
        result = convert_max_config_value(
            config_class=TestConfig,
            value="PAGED",
            field_type=KVCacheStrategy,
            field_name="cache_strategy",
        )
        assert result == KVCacheStrategy.PAGED

        result = convert_max_config_value(
            config_class=TestConfig,
            value="model_default",
            field_type=KVCacheStrategy,
            field_name="cache_strategy",
        )
        assert result == KVCacheStrategy.MODEL_DEFAULT

    def test_enum_conversion_dtype(self) -> None:
        """Test conversion to DType enum."""
        # Test common dtypes by string.
        result = convert_max_config_value(TestConfig, "float32", DType, "dtype")
        assert result == DType.float32

        result = convert_max_config_value(TestConfig, "float16", DType, "dtype")
        assert result == DType.float16

        result = convert_max_config_value(TestConfig, "int8", DType, "dtype")
        assert result == DType.int8

        result = convert_max_config_value(
            TestConfig, "bfloat16", DType, "dtype"
        )
        assert result == DType.bfloat16

        # Test with existing DType value.
        result = convert_max_config_value(
            TestConfig, DType.float32, DType, "dtype"
        )
        assert result == DType.float32

    def test_enum_conversion_invalid_values(self) -> None:
        """Test enum conversion with invalid values."""
        # Test invalid GPUProfilingMode.
        with pytest.raises(ValueError):
            convert_max_config_value(
                TestConfig, "invalid_mode", GPUProfilingMode, "gpu_profiling"
            )

        # Test invalid KVCacheStrategy.
        with pytest.raises(ValueError):
            convert_max_config_value(
                TestConfig,
                "invalid_strategy",
                KVCacheStrategy,
                "cache_strategy",
            )

    def test_list_type_conversion(self) -> None:
        """Test conversion of list types."""
        # Test list of strings.
        result = convert_max_config_value(
            TestConfig, ["path1", "path2", "path3"], list[str], "lora_paths"
        )
        assert result == ["path1", "path2", "path3"]

        # Test list of integers.
        result = convert_max_config_value(
            TestConfig, ["1", "2", "3"], list[int], "int_list"
        )
        assert result == [1, 2, 3]

        # Test list of booleans.
        result = convert_max_config_value(
            TestConfig, ["true", "false", "1", "0"], list[bool], "bool_list"
        )
        assert result == [True, False, True, False]

    def test_list_type_invalid_input(self) -> None:
        """Test list type conversion with invalid input."""
        # Test non-list input.
        with pytest.raises(ValueError, match="Expected list"):
            convert_max_config_value(
                TestConfig, "not_a_list", list[str], "test_field"
            )

    def test_optional_type_conversion(self) -> None:
        """Test conversion of Optional types."""
        # Test Optional[int] with valid value.
        result = convert_max_config_value(
            TestConfig, "42", Optional[int], "optional_int"
        )
        assert result == 42

        # Test Optional[int] with None.
        result = convert_max_config_value(
            TestConfig, None, Optional[int], "optional_int"
        )
        assert result is None

        # Test Optional[bool] with valid value.
        result = convert_max_config_value(
            TestConfig, "true", Optional[bool], "optional_bool"
        )
        assert result is True

        # Test Optional[bool] with None.
        result = convert_max_config_value(
            TestConfig, None, Optional[bool], "optional_bool"
        )
        assert result is None

    def test_union_type_conversion(self) -> None:
        """Test conversion of Union types."""
        # Test Union[int, str] with int.
        result = convert_max_config_value(
            TestConfig, "42", Union[int, str], "union_field"
        )
        assert result == 42

        # Test Union[int, str] with string that can't be converted to int.
        result = convert_max_config_value(
            TestConfig, "hello", Union[int, str], "union_field"
        )
        assert result == "hello"

        # Test Union[bool, str] with boolean string.
        result = convert_max_config_value(
            TestConfig, "true", Union[bool, str], "union_field"
        )
        assert result is True

    def test_complex_nested_types(self) -> None:
        """Test conversion of complex nested types."""
        # Test Optional[list[str]].
        result = convert_max_config_value(
            TestConfig, ["item1", "item2"], Optional[list[str]], "optional_list"
        )
        assert result == ["item1", "item2"]

        # Test Optional[list[str]] with None.
        result = convert_max_config_value(
            TestConfig, None, Optional[list[str]], "optional_list"
        )
        assert result is None

    def test_direct_type_instantiation_fallback(self) -> None:
        """Test fallback to direct type instantiation."""
        # Test with a simple type that should work with direct instantiation.
        result = convert_max_config_value(TestConfig, "3.14159", float, "pi")
        assert result == 3.14159

        # Test with integer.
        result = convert_max_config_value(TestConfig, "42", int, "answer")
        assert result == 42


class TestMAXConfigCLIArgParsers:
    """Test suite for MAXConfig cli_arg_parsers functionality."""

    def test_cli_arg_parsers_vs_config_file_basic(self) -> None:
        """Test cli_arg_parsers produces same result as CLI parsing."""
        # Create a config file with test values
        config_data = {
            "test_field": "from_config",
            "test_int": 42,
            "test_bool": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()

            # Load config from file
            config = TestConfig.from_config_file(f.name)
            config_file_parser = config.cli_arg_parsers()

            # Manually create CLI parser and parse equivalent args
            parser = argparse.ArgumentParser()
            parser.add_argument("--test-field", type=str, default="from_config")
            parser.add_argument("--test-int", type=int, default=42)
            parser.add_argument(
                "--test-bool", action="store_true", default=True
            )

            # Simulate CLI args that would produce same values
            cli_args = [
                "--test-field",
                "from_config",
                "--test-int",
                "42",
                "--test-bool",
            ]
            config_namespace = config_file_parser.parse_args(cli_args)
            cli_namespace = parser.parse_args(cli_args)

            # Both should produce identical namespaces
            assert (
                config_namespace.test_field
                == cli_namespace.test_field
                == "from_config"
            )
            assert config_namespace.test_int == cli_namespace.test_int == 42
            assert config_namespace.test_bool == cli_namespace.test_bool is True

    def test_cli_arg_parsers_vs_config_file_kv_cache(self) -> None:
        """Test cli_arg_parsers with KV cache config vs CLI parsing."""
        # Create a config file with KV cache values
        config_data = {
            "name": "test_kv_config",
            "kv_cache_config": {
                "cache_strategy": "paged",
                "kv_cache_page_size": 256,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()

            # Load config from file
            config = KVCacheConfig.from_config_file(f.name)
            parser = config.cli_arg_parsers()
            config_namespace = parser.parse_args([])

            assert config_namespace.cache_strategy == KVCacheStrategy.PAGED
            assert config_namespace.kv_cache_page_size == 256

            # Verify internal fields are excluded
            assert not hasattr(config_namespace, "_config_file_section_name")

    def test_cli_arg_parsers_vs_config_file_profiling(self) -> None:
        """Test cli_arg_parsers with profiling config vs CLI parsing."""
        # Create a config file with profiling values
        config_data = {
            "name": "test_profiling_config",
            "profiling_config": {
                "gpu_profiling": "detailed",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()

            # Load config from file
            config = ProfilingConfig.from_config_file(f.name)
            parser = config.cli_arg_parsers()
            config_namespace = parser.parse_args([])
            assert config_namespace.gpu_profiling == GPUProfilingMode.DETAILED

    @pytest.mark.skip(
        reason="TODO: Skipping LoRA config test for now - there's an issue with list[str] type mismatch for lora_paths"
    )
    def test_cli_arg_parsers_vs_config_file_lora(self) -> None:
        """Test cli_arg_parsers with LoRA config vs CLI parsing."""
        # Create a config file with LoRA values
        config_data = {
            "name": "test_lora_config",
            "lora_config": {
                "enable_lora": True,
                "lora_paths": ["model1.lora", "model2.lora"],
                "max_lora_rank": 64,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()

            # Load config from file
            config = LoRAConfig.from_config_file(f.name)
            parser = config.cli_arg_parsers()

            # Manually create CLI parser for LoRA args
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--enable-lora", action="store_true", default=False
            )
            parser.add_argument("--lora-paths", nargs="*", default=[])
            parser.add_argument("--max-lora-rank", type=int, default=16)

            # Parse CLI args that match config values
            cli_args = [
                "--enable-lora",
                "--lora-paths",
                "model1.lora",
                "model2.lora",
                "--max-lora-rank",
                "64",
            ]
            config_namespace = parser.parse_args(cli_args)
            cli_namespace = parser.parse_args(cli_args)

            # Both should produce identical values
            assert (
                config_namespace.enable_lora
                == cli_namespace.enable_lora
                is True
            )
            assert (
                config_namespace.lora_paths
                == cli_namespace.lora_paths
                == ["model1.lora", "model2.lora"]
            )
            assert (
                config_namespace.max_lora_rank
                == cli_namespace.max_lora_rank
                == 64
            )

    def test_cli_arg_parsers_vs_config_file_sampling(self) -> None:
        """Test cli_arg_parsers with sampling config vs CLI parsing."""
        # Create a config file with sampling values
        config_data = {
            "name": "test_sampling_config",
            "sampling_config": {
                "in_dtype": "float16",
                "out_dtype": "bfloat16",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()

            # Load config from file
            config = SamplingConfig.from_config_file(f.name)
            parser = config.cli_arg_parsers()
            config_namespace = parser.parse_args([])

            assert config_namespace.in_dtype == DType.float16
            assert config_namespace.out_dtype == DType.bfloat16

    def test_cli_arg_parsers_with_defaults(self) -> None:
        """Test cli_arg_parsers preserves defaults when no CLI args provided."""
        # Create default config
        config = TestConfig()  # Using all defaults
        parser = config.cli_arg_parsers()

        # Manually create CLI parser with same defaults
        parser = argparse.ArgumentParser()
        parser.add_argument("--test-field", type=str, default="default_value")
        parser.add_argument("--test-int", type=int, default=42)
        parser.add_argument("--test-bool", action="store_true", default=True)

        # Parse empty CLI args (all defaults)
        cli_args: list[str] = []
        config_namespace = parser.parse_args(cli_args)
        cli_namespace = parser.parse_args(cli_args)

        # Both should have identical default values
        assert (
            config_namespace.test_field
            == cli_namespace.test_field
            == "default_value"
        )
        assert config_namespace.test_int == cli_namespace.test_int == 42
        assert config_namespace.test_bool == cli_namespace.test_bool is True

    def test_cli_arg_parsers_field_name_conversion(self) -> None:
        """Test that field names use Python naming in namespace."""
        config = KVCacheConfig()
        parser = config.cli_arg_parsers()
        namespace = parser.parse_args([])

        # Field names should use underscores (Python convention)
        assert hasattr(namespace, "cache_strategy")
        assert hasattr(namespace, "kv_cache_page_size")

        # CLI-style hyphenated names should NOT exist
        assert not hasattr(namespace, "cache-strategy")
        assert not hasattr(namespace, "kv-cache-page-size")

    def test_cli_arg_parsers_standalone_script_pattern(self) -> None:
        """Test the standalone script usage pattern."""
        # Test the real-world pattern: config.cli_arg_parsers() for script usage
        configs = [
            KVCacheConfig(),
            SamplingConfig(),
            ProfilingConfig(),
            LoRAConfig(),
        ]

        for config in configs:
            parser = config.cli_arg_parsers()
            namespace = parser.parse_args([])

            # Should be proper argparse.Namespace
            assert isinstance(namespace, argparse.Namespace)

            # Should have public fields accessible
            public_fields = [
                f.name for f in fields(config) if not f.name.startswith("_")
            ]
            for field_name in public_fields:
                assert hasattr(namespace, field_name)

            # Should exclude internal fields
            assert not hasattr(namespace, "_config_file_section_name")

    def test_cli_arg_parsers_required_params_basic(self) -> None:
        """Test cli_arg_parsers with required_params parameter."""
        config = TestConfig()

        # Test with no required params (default behavior)
        parser = config.cli_arg_parsers()
        namespace = parser.parse_args([])
        assert namespace.test_field == "default_value"
        assert namespace.test_int == 42

        # Test with required params - should fail when required field not provided
        parser_with_required = config.cli_arg_parsers(
            required_params={"test_field"}
        )

        # Should work when required field is provided
        namespace = parser_with_required.parse_args(
            ["--test-field", "provided_value"]
        )
        assert namespace.test_field == "provided_value"

        # Should fail when required field is not provided
        with pytest.raises(
            SystemExit
        ):  # argparse raises SystemExit on missing required args
            parser_with_required.parse_args([])

    def test_cli_arg_parsers_required_params_multiple(self) -> None:
        """Test cli_arg_parsers with multiple required parameters."""
        config = TestConfig()
        parser = config.cli_arg_parsers(
            required_params={"test_field", "test_int"}
        )

        # Should work when all required fields are provided
        namespace = parser.parse_args(
            ["--test-field", "required_value", "--test-int", "100"]
        )
        assert namespace.test_field == "required_value"
        assert namespace.test_int == 100
        assert (
            namespace.test_bool is True
        )  # Default value for non-required field

        # Should fail when only some required fields are provided
        with pytest.raises(SystemExit):
            parser.parse_args(["--test-field", "value"])  # Missing test_int

        with pytest.raises(SystemExit):
            parser.parse_args(["--test-int", "100"])  # Missing test_field

    def test_cli_arg_parsers_required_params_with_choices(self) -> None:
        """Test cli_arg_parsers with both required_params and choices_provider."""
        config = TestConfig()
        choices = {"test_field": ["option1", "option2", "option3"]}

        parser = config.cli_arg_parsers(
            choices_provider=choices, required_params={"test_field"}
        )

        # Should work with valid choice
        namespace = parser.parse_args(["--test-field", "option1"])
        assert namespace.test_field == "option1"

        # Should fail with invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--test-field", "invalid_option"])

        # Should fail when required field not provided
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_cli_arg_parsers_required_params_empty_set(self) -> None:
        """Test cli_arg_parsers with empty required_params set."""
        config = TestConfig()

        # Explicit empty set should behave same as None/default
        parser_empty = config.cli_arg_parsers(required_params=set())
        parser_default = config.cli_arg_parsers()

        # Both should work with no args (all defaults)
        namespace_empty = parser_empty.parse_args([])
        namespace_default = parser_default.parse_args([])

        assert namespace_empty.test_field == namespace_default.test_field
        assert namespace_empty.test_int == namespace_default.test_int
        assert namespace_empty.test_bool == namespace_default.test_bool

    def test_cli_arg_parsers_required_params_nonexistent_field(self) -> None:
        """Test cli_arg_parsers with required_params containing non-existent field."""
        config = TestConfig()

        # Should not crash when required_params contains non-existent field
        # The field simply won't exist in the parser, so no requirement is added
        parser = config.cli_arg_parsers(
            required_params={"nonexistent_field", "test_field"}
        )

        # Should still require the existing field
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing test_field

        # Should work when existing required field is provided
        namespace = parser.parse_args(["--test-field", "value"])
        assert namespace.test_field == "value"

    def test_cli_arg_parsers_required_params_with_kv_cache_config(self) -> None:
        """Test cli_arg_parsers required_params with real config class."""
        config = KVCacheConfig()

        # Test requiring cache_strategy
        parser = config.cli_arg_parsers(required_params={"cache_strategy"})

        # Should work when required field is provided
        namespace = parser.parse_args(["--cache-strategy", "paged"])
        assert namespace.cache_strategy == KVCacheStrategy.PAGED

        # Should fail when required field not provided
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_cli_arg_parsers_required_params_boolean_fields(self) -> None:
        """Test cli_arg_parsers required_params with boolean fields."""
        config = TestConfig()

        # Test requiring a boolean field
        parser = config.cli_arg_parsers(required_params={"test_bool"})

        # Boolean fields with BooleanOptionalAction should work
        namespace = parser.parse_args(["--test-bool"])
        assert namespace.test_bool is True

        namespace = parser.parse_args(["--no-test-bool"])
        assert namespace.test_bool is False

        # Should fail when boolean field not provided
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_cli_arg_parsers_union_syntax_support(self) -> None:
        """Test that cli_arg_parsers handles both Optional[T] and T | None syntaxes."""

        @dataclass
        class UnionTestConfig(MAXConfig):
            _config_file_section_name: str = "union_test_config"

            # Test both union syntaxes
            optional_int_old: Optional[int] = None
            optional_int_new: int | None = None
            optional_str_old: Optional[str] = None
            optional_str_new: str | None = None

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "optional_int_old": "Optional int using Optional[int] syntax",
                    "optional_int_new": "Optional int using int | None syntax",
                    "optional_str_old": "Optional str using Optional[str] syntax",
                    "optional_str_new": "Optional str using str | None syntax",
                }

        config = UnionTestConfig()
        parser = config.cli_arg_parsers()

        # Test that both syntaxes produce integer types for integer fields
        args = parser.parse_args(
            [
                "--optional-int-old",
                "42",
                "--optional-int-new",
                "84",
                "--optional-str-old",
                "hello",
                "--optional-str-new",
                "world",
            ]
        )

        # Both integer fields should be parsed as integers, not strings
        assert isinstance(args.optional_int_old, int)
        assert args.optional_int_old == 42
        assert isinstance(args.optional_int_new, int)
        assert args.optional_int_new == 84

        # String fields should remain as strings
        assert isinstance(args.optional_str_old, str)
        assert args.optional_str_old == "hello"
        assert isinstance(args.optional_str_new, str)
        assert args.optional_str_new == "world"

        # Test arithmetic operations work (this was the original bug)
        result_old = args.optional_int_old - 1
        result_new = args.optional_int_new - 1
        assert result_old == 41
        assert result_new == 83

        # Test with no arguments (should use defaults)
        args_default = parser.parse_args([])
        assert args_default.optional_int_old is None
        assert args_default.optional_int_new is None
        assert args_default.optional_str_old is None
        assert args_default.optional_str_new is None

    def test_get_default_required_fields(self) -> None:
        """Test get_default_required_fields static method."""
        # Test that base MAXConfig returns empty set
        assert MAXConfig.get_default_required_fields() == set()

        # Test that concrete configs return empty set by default
        assert TestConfig.get_default_required_fields() == set()
        assert KVCacheConfig.get_default_required_fields() == set()
        assert SamplingConfig.get_default_required_fields() == set()
        assert ProfilingConfig.get_default_required_fields() == set()
        assert LoRAConfig.get_default_required_fields() == set()

    def test_cli_arg_parsers_uses_get_default_required_fields(self) -> None:
        """Test that cli_arg_parsers uses get_default_required_fields when required_params is None."""

        # Create a test config that overrides get_default_required_fields
        @dataclass
        class ConfigWithDefaultRequired(MAXConfig):
            _config_file_section_name: str = "test_config"
            test_field: str = "default"
            other_field: str = "other_default"

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "test_field": "A test field",
                    "other_field": "Another test field",
                }

            @classmethod
            def get_default_required_fields(cls) -> set[str]:
                return {"test_field"}

        config = ConfigWithDefaultRequired()

        # When required_params is None, should use get_default_required_fields
        parser = config.cli_arg_parsers()

        # Should require test_field (from get_default_required_fields)
        with pytest.raises(SystemExit):
            parser.parse_args([])

        # Should work when required field is provided
        namespace = parser.parse_args(["--test-field", "provided"])
        assert namespace.test_field == "provided"
        assert namespace.other_field == "other_default"

        # Explicit required_params should override get_default_required_fields
        parser_override = config.cli_arg_parsers(
            required_params={"other_field"}
        )

        # Should require other_field instead of test_field
        with pytest.raises(SystemExit):
            parser_override.parse_args(
                ["--test-field", "provided"]
            )  # Missing other_field

        # Should work when overridden required field is provided
        namespace = parser_override.parse_args(["--other-field", "provided"])
        assert namespace.other_field == "provided"
        assert namespace.test_field == "default"

    def test_cli_arg_parsers_formatter_class(self) -> None:
        """Test cli_arg_parsers with custom formatter class."""
        config = TestConfig()

        # Test with RawDescriptionHelpFormatter
        parser = config.cli_arg_parsers(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Test description\nwith multiple lines",
        )

        # Verify the parser was created with the custom formatter
        assert parser.formatter_class is argparse.RawDescriptionHelpFormatter

        # Test that the description is preserved (RawDescriptionHelpFormatter preserves newlines)
        help_text = parser.format_help()
        assert "Test description" in help_text
        assert "with multiple lines" in help_text

        # Test with ArgumentDefaultsHelpFormatter
        parser_with_defaults = config.cli_arg_parsers(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Verify the parser was created with the custom formatter
        assert (
            parser_with_defaults.formatter_class
            is argparse.ArgumentDefaultsHelpFormatter
        )

        # Test that parsing still works with custom formatter
        namespace = parser_with_defaults.parse_args(
            ["--test-field", "custom_value"]
        )
        assert namespace.test_field == "custom_value"
        assert namespace.test_int == 42  # Default value from TestConfig
        assert namespace.test_bool is True  # Default value from TestConfig

        # Test with custom formatter class that has max_help_position=80
        class CustomRawTextHelpFormatter(argparse.RawTextHelpFormatter):
            def __init__(self, prog: str) -> None:
                super().__init__(prog, max_help_position=80)

        parser_custom = config.cli_arg_parsers(
            formatter_class=CustomRawTextHelpFormatter
        )

        # Verify the parser was created with the custom formatter
        assert parser_custom.formatter_class is CustomRawTextHelpFormatter

        # Test that parsing still works with custom formatter
        namespace_custom = parser_custom.parse_args(["--test-int", "43"])
        assert namespace_custom.test_int == 43
        assert (
            namespace_custom.test_field == "default_value"
        )  # Default value from TestConfig


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


class TestMAXConfigArgumentGroups:
    """Test suite for MAXConfig argument group functionality."""

    def test_cli_arg_parsers_with_groups_basic(self) -> None:
        """Test cli_arg_parsers with argument grouping."""

        # Create a test config with groups
        @dataclass
        class TestConfigWithGroups(MAXConfig):
            _config_file_section_name: str = "test_config_with_groups"

            # Group 1
            field1: str = field(
                default="default1",
                metadata={
                    "group": "Group 1",
                    "group_description": "First group of settings",
                },
            )
            field2: int = field(default=42, metadata={"group": "Group 1"})

            # Group 2
            field3: bool = field(
                default=False,
                metadata={
                    "group": "Group 2",
                    "group_description": "Second group of settings",
                },
            )

            # Ungrouped
            field4: str = "ungrouped"

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "field1": "First field in group 1",
                    "field2": "Second field in group 1",
                    "field3": "First field in group 2",
                    "field4": "Ungrouped field",
                }

        config = TestConfigWithGroups()

        parser_with_groups = config.cli_arg_parsers()

        # Check that argument groups were created
        assert (
            len(parser_with_groups._action_groups) >= 2
        )  # At least 2 groups + main group

        # Find our custom groups
        group1 = None
        group2 = None
        for group in parser_with_groups._action_groups:
            if group.title == "Group 1":
                group1 = group
            elif group.title == "Group 2":
                group2 = group

        assert group1 is not None, "Group 1 should be created"
        assert group2 is not None, "Group 2 should be created"

        # Check that group descriptions are set
        assert group1.description == "First group of settings"
        assert group2.description == "Second group of settings"

        # Test parsing still works
        args = parser_with_groups.parse_args(
            [
                "--field1",
                "test_value",
                "--field2",
                "100",
                "--field3",
                "--field4",
                "custom_value",
            ]
        )

        assert args.field1 == "test_value"
        assert args.field2 == 100
        assert args.field3 is True
        assert args.field4 == "custom_value"

    def test_cli_arg_parsers_ungrouped_fields_only(self) -> None:
        """Test cli_arg_parsers with only ungrouped fields."""

        @dataclass
        class TestConfigUngrouped(MAXConfig):
            _config_file_section_name: str = "test_config_ungrouped"

            field1: str = "default1"
            field2: int = 42
            field3: bool = False

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "field1": "Field 1",
                    "field2": "Field 2",
                    "field3": "Field 3",
                }

        config = TestConfigUngrouped()

        # Test with only ungrouped fields
        parser = config.cli_arg_parsers()

        # Should have at least the main group (argparse always creates at least one group)
        assert len(parser._action_groups) >= 1

        # Test parsing still works
        args = parser.parse_args(
            ["--field1", "test_value", "--field2", "100", "--field3"]
        )

        assert args.field1 == "test_value"
        assert args.field2 == 100
        assert args.field3 is True

    def test_cli_arg_parsers_mixed_groups_and_ungrouped(self) -> None:
        """Test cli_arg_parsers with both grouped and ungrouped fields."""

        @dataclass
        class TestConfigMixed(MAXConfig):
            _config_file_section_name: str = "test_config_mixed"

            # Grouped fields
            grouped_field1: str = field(
                default="grouped1", metadata={"group": "Grouped Settings"}
            )
            grouped_field2: int = field(
                default=10, metadata={"group": "Grouped Settings"}
            )

            # Ungrouped fields
            ungrouped_field1: str = "ungrouped1"
            ungrouped_field2: bool = False

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "grouped_field1": "Grouped field 1",
                    "grouped_field2": "Grouped field 2",
                    "ungrouped_field1": "Ungrouped field 1",
                    "ungrouped_field2": "Ungrouped field 2",
                }

        config = TestConfigMixed()
        parser = config.cli_arg_parsers()

        # Should have at least 2 groups: main group + our custom group
        assert len(parser._action_groups) >= 2

        # Find our custom group
        custom_group = None
        for group in parser._action_groups:
            if group.title == "Grouped Settings":
                custom_group = group
                break

        assert custom_group is not None, "Custom group should be created"

        # Test parsing works for both grouped and ungrouped fields
        args = parser.parse_args(
            [
                "--grouped-field1",
                "test1",
                "--grouped-field2",
                "20",
                "--ungrouped-field1",
                "test2",
                "--ungrouped-field2",
            ]
        )

        assert args.grouped_field1 == "test1"
        assert args.grouped_field2 == 20
        assert args.ungrouped_field1 == "test2"
        assert args.ungrouped_field2 is True

    def test_cli_arg_parsers_group_description_extraction(self) -> None:
        """Test that group descriptions are properly extracted from field metadata."""

        @dataclass
        class TestConfigDescription(MAXConfig):
            _config_file_section_name: str = "test_config_description"

            # Field with group_description
            field1: str = field(
                default="default1",
                metadata={
                    "group": "Test Group",
                    "group_description": "This is a test group description",
                },
            )

            # Field without group_description (should still be in same group)
            field2: int = field(default=42, metadata={"group": "Test Group"})

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "field1": "Field 1",
                    "field2": "Field 2",
                }

        config = TestConfigDescription()
        parser = config.cli_arg_parsers()

        # Find our test group
        test_group = None
        for group in parser._action_groups:
            if group.title == "Test Group":
                test_group = group
                break

        assert test_group is not None, "Test group should be created"
        assert test_group.description == "This is a test group description"

    def test_cli_arg_parsers_existing_configs(self) -> None:
        """Test that existing configs without groups still work."""
        # Use existing TestConfig from the file
        config = TestConfig()

        # Test that existing configs work (ungrouped fields go to main parser)
        parser = config.cli_arg_parsers()
        args = parser.parse_args(
            ["--test-field", "test_value", "--test-int", "100", "--test-bool"]
        )

        assert args.test_field == "test_value"
        assert args.test_int == 100
        assert args.test_bool is True

    def test_cli_arg_parsers_enum_conversion_with_groups(self) -> None:
        """Test that enum conversion still works with argument groups."""

        @dataclass
        class TestConfigEnum(MAXConfig):
            _config_file_section_name: str = "test_config_enum"

            enum_field: GPUProfilingMode = field(
                default=GPUProfilingMode.OFF,
                metadata={"group": "Enum Settings"},
            )

            @classmethod
            def _get_enum_mapping_impl(cls):
                return {"GPUProfilingMode": GPUProfilingMode}

            @staticmethod
            def help() -> dict[str, str]:
                return {"enum_field": "Enum field test"}

        config = TestConfigEnum()
        parser = config.cli_arg_parsers()

        # Test parsing with enum value
        args = parser.parse_args(["--enum-field", "detailed"])

        # Should be converted to proper enum object
        assert args.enum_field == GPUProfilingMode.DETAILED
        assert isinstance(args.enum_field, GPUProfilingMode)
