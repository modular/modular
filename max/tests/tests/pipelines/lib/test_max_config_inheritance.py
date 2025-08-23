# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for MAXConfig inheritance functionality."""

from __future__ import annotations

import enum
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import pytest
import yaml
from max.pipelines.lib import (
    MAXConfig,
    deep_merge_max_configs,
    resolve_max_config_inheritance,
)


# Test enums for enum inheritance tests
class TestEnum1(enum.Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"


class TestEnum2(enum.Enum):
    VALUE3 = "value3"
    VALUE4 = "value4"


class TestEnum3(enum.Enum):
    VALUE5 = "value5"
    VALUE6 = "value6"


@dataclass
class BaseTestConfig(MAXConfig):
    """Base test config with some enums."""

    _config_file_section_name: str = "base_test_config"

    @staticmethod
    def help() -> dict[str, str]:
        return {}

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for BaseTestConfig."""
        return {
            "TestEnum1": TestEnum1,
            "TestEnum2": TestEnum2,
        }


@dataclass
class DerivedTestConfig(BaseTestConfig):
    """Derived test config that adds more enums."""

    _config_file_section_name: str = "derived_test_config"

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for DerivedTestConfig."""
        return {
            "TestEnum3": TestEnum3,
        }


@dataclass
class OverrideTestConfig(BaseTestConfig):
    """Test config that overrides a parent enum."""

    _config_file_section_name: str = "override_test_config"

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for OverrideTestConfig, overriding TestEnum1."""
        return {
            "TestEnum1": TestEnum3,  # Override TestEnum1 with TestEnum3
        }


@dataclass
class TestConfig(MAXConfig):
    """Test config class for unit testing."""

    _config_file_section_name: str = "test_config"

    test_field: str = "default_value"
    test_int: int = 42
    test_bool: bool = True

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "test_field": "A test string field",
            "test_int": "A test integer field",
            "test_bool": "A test boolean field",
        }


class TestMAXConfigInheritance:
    """Test suite for configuration inheritance functionality."""

    def test_simple_inheritance(self) -> None:
        """Test basic inheritance from a base config."""
        # Create base config file.
        base_config_data = {
            "name": "base_config",
            "version": "1.0",
            "test_config": {
                "test_field": "base_value",
                "test_int": 100,
                "test_bool": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as base_f:
            yaml.dump(base_config_data, base_f)
            base_f.flush()  # Ensure data is written to disk
            base_config_path = base_f.name

            # Create child config file that inherits from base.
            child_config_data = {
                "name": "child_config",
                "version": "1.0",
                "depends_on": base_config_path,
                "test_config": {
                    "test_field": "child_value",  # Override base value
                    # test_int and test_bool should be inherited from base
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as child_f:
                yaml.dump(child_config_data, child_f)
                child_f.flush()  # Ensure data is written to disk
                child_config_path = child_f.name

                config = TestConfig.from_config_file(child_config_path)
                assert config.test_field == "child_value"  # Overridden value
                assert config.test_int == 100  # Inherited from base
                assert config.test_bool is False  # Inherited from base

    def test_chained_inheritance(self) -> None:
        """Test inheritance chain: grandparent -> parent -> child."""
        # Create grandparent config.
        grandparent_config_data = {
            "name": "grandparent_config",
            "version": "1.0",
            "test_config": {
                "test_field": "grandparent_value",
                "test_int": 1,
                "test_bool": True,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml"
        ) as grandparent_f:
            yaml.dump(grandparent_config_data, grandparent_f)
            grandparent_f.flush()  # Ensure data is written to disk
            grandparent_config_path = grandparent_f.name

            # Create parent config that inherits from grandparent.
            parent_config_data = {
                "name": "parent_config",
                "version": "1.0",
                "depends_on": grandparent_config_path,
                "test_config": {
                    "test_field": "parent_value",  # Override grandparent
                    "test_int": 2,  # Override grandparent
                    # test_bool inherited from grandparent
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as parent_f:
                yaml.dump(parent_config_data, parent_f)
                parent_f.flush()  # Ensure data is written to disk
                parent_config_path = parent_f.name

                # Create child config that inherits from parent.
                child_config_data = {
                    "name": "child_config",
                    "version": "1.0",
                    "depends_on": parent_config_path,
                    "test_config": {
                        "test_field": "child_value",  # Override parent
                        # test_int inherited from parent, test_bool inherited from grandparent
                    },
                }

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml"
                ) as child_f:
                    yaml.dump(child_config_data, child_f)
                    child_f.flush()  # Ensure data is written to disk
                    child_config_path = child_f.name

                    config = TestConfig.from_config_file(child_config_path)
                    assert config.test_field == "child_value"  # Final override
                    assert config.test_int == 2  # From parent
                    assert config.test_bool is True  # From grandparent

    def test_inheritance_base_file_not_found(self) -> None:
        """Test that missing base config file raises ValueError."""
        child_config_data = {
            "name": "child_config",
            "version": "1.0",
            "depends_on": "/path/to/nonexistent/config.yaml",
            "test_config": {
                "test_field": "child_value",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as child_f:
            yaml.dump(child_config_data, child_f)
            child_f.flush()  # Ensure data is written to disk
            child_config_path = child_f.name

            # Should now raise ValueError instead of falling back gracefully
            with pytest.raises(
                FileNotFoundError, match="Base configuration file not found"
            ):
                TestConfig.from_config_file(child_config_path)

    def test_inheritance_base_config_invalid_yaml(self) -> None:
        """Test graceful handling when base config has invalid YAML."""
        # Create base config with invalid YAML.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as base_f:
            base_f.write("invalid: yaml: content: [")
            base_f.flush()  # Ensure data is written to disk
            base_config_path = base_f.name

            child_config_data = {
                "name": "child_config",
                "version": "1.0",
                "depends_on": base_config_path,
                "test_config": {
                    "test_field": "child_value",
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as child_f:
                yaml.dump(child_config_data, child_f)
                child_f.flush()  # Ensure data is written to disk
                child_config_path = child_f.name

                # Should fall back to child config only and log warning.
                config = TestConfig.from_config_file(child_config_path)
                assert config.test_field == "child_value"
                assert (
                    config.test_int == 42
                )  # Default value since inheritance failed
                assert (
                    config.test_bool is True
                )  # Default value since inheritance failed

    def test_inheritance_base_config_not_dict(self) -> None:
        """Test error when base config is not a dictionary."""
        # Create base config that's a list instead of dict.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as base_f:
            yaml.dump(["item1", "item2"], base_f)
            base_f.flush()  # Ensure data is written to disk
            base_config_path = base_f.name

            child_config_data = {
                "name": "child_config",
                "version": "1.0",
                "depends_on": base_config_path,  # TOP LEVEL inheritance
                "test_config": {
                    "test_field": "child_value",
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as child_f:
                yaml.dump(child_config_data, child_f)
                child_f.flush()  # Ensure data is written to disk
                child_config_path = child_f.name

                # Should now raise ValueError for invalid base config structure
                with pytest.raises(
                    ValueError,
                    match="must contain a dictionary at the top level",
                ):
                    TestConfig.from_config_file(child_config_path)

    def test_inheritance_no_depends_on(self) -> None:
        """Test that configs without depends_on work normally."""
        config_data = {
            "test_field": "normal_value",
            "test_int": 200,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config_data, f)
            f.flush()  # Ensure data is written to disk
            config_path = f.name

            config = TestConfig.from_config_file(config_path)
            assert config.test_field == "normal_value"
            assert config.test_int == 200
            assert config.test_bool is True  # Default value

    def test_inheritance_comprehensive_config_file(self) -> None:
        """Test inheritance with comprehensive config files (multiple sections)."""
        # Create base comprehensive config
        base_config_data = {
            "name": "base_config",
            "version": "1.0",
            "test_config": {
                "test_field": "base_value",
                "test_int": 100,
                "test_bool": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as base_f:
            yaml.dump(base_config_data, base_f)
            base_f.flush()  # Ensure data is written to disk
            base_config_path = base_f.name

            # Create child comprehensive config that inherits from base
            child_config_data = {
                "name": "child_config",
                "version": "2.0",
                "depends_on": base_config_path,  # TOP LEVEL inheritance
                "test_config": {
                    "test_field": "child_value",  # Override base value
                    # test_int and test_bool should be inherited from base
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as child_f:
                yaml.dump(child_config_data, child_f)
                child_f.flush()  # Ensure data is written to disk
                child_config_path = child_f.name

                config = TestConfig.from_config_file(child_config_path)
                assert config.test_field == "child_value"  # Overridden value
                assert config.test_int == 100  # Inherited from base
                assert config.test_bool is False  # Inherited from base

    def test_resolve_inheritance_missing_file_error(self) -> None:
        """Test that resolve_max_config_inheritance raises ValueError for missing files."""
        config_dict = {
            "depends_on": "/path/to/nonexistent/config.yaml",
            "test_field": "value",
        }

        with pytest.raises(
            FileNotFoundError, match="Base configuration file not found"
        ):
            resolve_max_config_inheritance(config_dict, TestConfig)

    def test_resolve_inheritance_no_depends_on(self) -> None:
        """Test that resolve_max_config_inheritance returns config unchanged when no depends_on."""
        config_dict = {"test_field": "value", "test_int": 100}

        result = resolve_max_config_inheritance(config_dict, TestConfig)
        assert result == config_dict  # Should return unchanged

    def test_deep_merge_configs(self) -> None:
        """Test that _deep_merge_configs correctly merges nested dictionaries."""
        base_config = {
            "name": "base",
            "version": "1.0",
            "test_config": {
                "test_field": "base_value",
                "test_int": 100,
                "test_bool": False,
            },
            "other_section": {
                "setting1": "base_setting1",
                "setting2": "base_setting2",
            },
        }

        child_config = {
            "name": "child",
            "test_config": {
                "test_field": "child_value",  # override
                # test_int and test_bool should be inherited
            },
            "other_section": {
                "setting1": "child_setting1",  # override
                # setting2 should be inherited
            },
            "new_section": {
                "new_setting": "new_value",
            },
        }

        result = deep_merge_max_configs(base_config, child_config)

        # Check top-level fields
        assert result["name"] == "child"  # Overridden
        assert result["version"] == "1.0"  # Inherited

        # Check test_config section
        assert (
            result["test_config"]["test_field"] == "child_value"
        )  # Overridden
        assert result["test_config"]["test_int"] == 100  # Inherited
        assert result["test_config"]["test_bool"] is False  # Inherited

        # Check other_section
        assert (
            result["other_section"]["setting1"] == "child_setting1"
        )  # Overridden
        assert (
            result["other_section"]["setting2"] == "base_setting2"
        )  # Inherited

        # Check new section
        assert result["new_section"]["new_setting"] == "new_value"  # Added

    def test_inheritance_with_union_syntax_fields(self) -> None:
        """Test that inheritance works correctly with both Optional[T] and T | None field types."""

        @dataclass
        class UnionInheritanceTestConfig(MAXConfig):
            _config_file_section_name: str = "union_inheritance_test_config"

            # Test both union syntaxes
            optional_int_old: Optional[int] = None
            optional_int_new: int | None = None
            required_str: str = "default"

            @staticmethod
            def help() -> dict[str, str]:
                return {
                    "optional_int_old": "Optional int using Optional[int] syntax",
                    "optional_int_new": "Optional int using int | None syntax",
                    "required_str": "Required string field",
                }

        # Create base config file with union syntax fields
        base_config_data = {
            "name": "base_config",
            "version": "1.0",
            "union_inheritance_test_config": {
                "optional_int_old": 100,
                "optional_int_new": 200,
                "required_str": "base_value",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as base_f:
            yaml.dump(base_config_data, base_f)
            base_f.flush()
            base_config_path = base_f.name

            # Create child config that inherits and overrides some values
            child_config_data = {
                "name": "child_config",
                "version": "1.0",
                "depends_on": base_config_path,
                "union_inheritance_test_config": {
                    "optional_int_old": 150,  # Override base value
                    # optional_int_new and required_str should be inherited from base
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as child_f:
                yaml.dump(child_config_data, child_f)
                child_f.flush()
                child_config_path = child_f.name

                # Load the config and verify inheritance worked
                config = UnionInheritanceTestConfig.from_config_file(
                    child_config_path
                )

                assert config.optional_int_old == 150  # Overridden value
                assert config.optional_int_new == 200  # Inherited from base
                assert (
                    config.required_str == "base_value"
                )  # Inherited from base

                # Test CLI argument parsing with inherited config
                parser = config.cli_arg_parsers()

                # Test that both union syntaxes work correctly after inheritance
                args = parser.parse_args(
                    ["--optional-int-old", "300", "--optional-int-new", "400"]
                )

                # Both should be parsed as integers, not strings
                assert isinstance(args.optional_int_old, int)
                assert args.optional_int_old == 300
                assert isinstance(args.optional_int_new, int)
                assert args.optional_int_new == 400

                # Test arithmetic operations (the original bug scenario)
                result_old = args.optional_int_old - 1
                result_new = args.optional_int_new - 1
                assert result_old == 299
                assert result_new == 399

    def test_inheritance_with_cli_override_base_values(self) -> None:
        """Test that CLI args can override base config values when child config doesn't define them."""
        # Create base config file with values
        base_config_data = {
            "name": "base_config",
            "version": "1.0",
            "test_config": {
                "test_field": "base_value",
                "test_int": 100,
                "test_bool": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as base_f:
            yaml.dump(base_config_data, base_f)
            base_f.flush()
            base_config_path = base_f.name

            # Create child config that inherits but doesn't define the test_config section
            child_config_data = {
                "name": "child_config",
                "version": "1.0",
                "depends_on": base_config_path,
                # No test_config section - should inherit all values from base
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml"
            ) as child_f:
                yaml.dump(child_config_data, child_f)
                child_f.flush()
                child_config_path = child_f.name

                # Load config from file first to verify inheritance
                config = TestConfig.from_config_file(child_config_path)
                assert config.test_field == "base_value"  # Inherited from base
                assert config.test_int == 100  # Inherited from base
                assert config.test_bool is False  # Inherited from base

                # Now test CLI arg parsing with the inherited config
                parser = config.cli_arg_parsers()

                # Override base values with CLI arguments
                args = parser.parse_args(
                    [
                        "--test-field",
                        "cli_override_value",
                        "--test-int",
                        "999",
                        "--test-bool",
                    ]
                )

                # Verify CLI args override the inherited base values
                assert args.test_field == "cli_override_value"
                assert args.test_int == 999
                assert args.test_bool is True

                # Test that the original config values are still intact
                assert config.test_field == "base_value"
                assert config.test_int == 100
                assert config.test_bool is False

    def test_base_config_enum_mapping(self) -> None:
        """Test that base config returns its own enums."""
        enum_mapping = BaseTestConfig._get_enum_mapping()

        assert "TestEnum1" in enum_mapping
        assert "TestEnum2" in enum_mapping
        assert enum_mapping["TestEnum1"] is TestEnum1
        assert enum_mapping["TestEnum2"] is TestEnum2
        assert len(enum_mapping) == 2

    def test_derived_config_enum_mapping(self) -> None:
        """Test that derived config returns union of parent and own enums."""
        enum_mapping = DerivedTestConfig._get_enum_mapping()

        # Should have enums from parent class
        assert "TestEnum1" in enum_mapping
        assert "TestEnum2" in enum_mapping
        assert enum_mapping["TestEnum1"] is TestEnum1
        assert enum_mapping["TestEnum2"] is TestEnum2

        # Should have enums from derived class
        assert "TestEnum3" in enum_mapping
        assert enum_mapping["TestEnum3"] is TestEnum3

        # Total count should be 3
        assert len(enum_mapping) == 3

    def test_override_config_enum_mapping(self) -> None:
        """Test that derived config can override parent enums."""
        enum_mapping = OverrideTestConfig._get_enum_mapping()

        # TestEnum1 should be overridden with TestEnum3
        assert "TestEnum1" in enum_mapping
        assert enum_mapping["TestEnum1"] is TestEnum3  # Overridden

        # TestEnum2 should still be from parent
        assert "TestEnum2" in enum_mapping
        assert enum_mapping["TestEnum2"] is TestEnum2

        # Total count should be 2
        assert len(enum_mapping) == 2

    def test_inheritance_chain_enum_mapping(self) -> None:
        """Test that enums are collected through the entire inheritance chain."""

        # Create a deeper inheritance chain
        @dataclass
        class DeepBaseConfig(MAXConfig):
            _config_file_section_name: str = "deep_base_config"

            @staticmethod
            def help() -> dict[str, str]:
                return {}

            @classmethod
            def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
                return {"DeepEnum1": TestEnum1}

        @dataclass
        class MiddleConfig(DeepBaseConfig):
            _config_file_section_name: str = "middle_config"

            @classmethod
            def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
                return {"MiddleEnum1": TestEnum2}

        @dataclass
        class TopConfig(MiddleConfig):
            _config_file_section_name: str = "top_config"

            @classmethod
            def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
                return {"TopEnum1": TestEnum3}

        enum_mapping = TopConfig._get_enum_mapping()

        # Should have enums from all levels
        assert "DeepEnum1" in enum_mapping
        assert "MiddleEnum1" in enum_mapping
        assert "TopEnum1" in enum_mapping

        assert enum_mapping["DeepEnum1"] is TestEnum1
        assert enum_mapping["MiddleEnum1"] is TestEnum2
        assert enum_mapping["TopEnum1"] is TestEnum3

        assert len(enum_mapping) == 3

    def test_empty_enum_mapping(self) -> None:
        """Test that configs with no enums work correctly."""

        @dataclass
        class EmptyConfig(MAXConfig):
            _config_file_section_name: str = "empty_config"

            @staticmethod
            def help() -> dict[str, str]:
                return {}

            @classmethod
            def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
                return {}

        enum_mapping = EmptyConfig._get_enum_mapping()
        assert enum_mapping == {}

    def test_missing_enum_mapping_impl(self) -> None:
        """Test that configs without _get_enum_mapping_impl work correctly."""

        @dataclass
        class NoImplConfig(MAXConfig):
            _config_file_section_name: str = "no_impl_config"

            @staticmethod
            def help() -> dict[str, str]:
                return {}

        enum_mapping = NoImplConfig._get_enum_mapping()
        assert enum_mapping == {}
