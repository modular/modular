# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for MAXConfig inheritance functionality."""

import tempfile
from dataclasses import dataclass

import pytest
import yaml
from max.pipelines.lib import (
    MAXConfig,
    deep_merge_max_configs,
    resolve_max_config_inheritance,
)


@dataclass
class TestConfig(MAXConfig):
    """Test config class for unit testing."""

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

    def test_inheritance_relative_path_not_supported(self) -> None:
        """Test that using relative path for inheritance raises ValueError."""
        child_config_data = {
            "name": "child_config",
            "version": "1.0",
            "depends_on": "relative/path/config.yaml",  # TOP LEVEL inheritance
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
                ValueError, match="Relative path inheritance not supported"
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

    def test_resolve_inheritance_relative_path_error(self) -> None:
        """Test that resolve_max_config_inheritance raises ValueError for relative paths."""
        config_dict = {
            "depends_on": "relative/path/config.yaml",
            "test_field": "value",
        }

        with pytest.raises(
            ValueError, match="Relative path inheritance not supported"
        ):
            resolve_max_config_inheritance(config_dict, TestConfig)

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
