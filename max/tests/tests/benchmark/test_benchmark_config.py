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
"""Benchmark config utility functions unit tests"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

# Import the utility functions from benchmark_shared.config
from max.benchmark.benchmark_shared.config import (
    BaseBenchmarkConfig,
    ServingBenchmarkConfig,
    SweepServingBenchmarkConfig,
    _add_config_file_arg_to_parser,
    _load_user_provided_config,
    _resolve_user_provided_config_file_cli_arg,
    parse_benchmark_args,
)


# Helper functions for creating test config files
def create_test_config_file(content: Any, suffix: str = ".yaml") -> Path:
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    ) as f:
        yaml.dump(content, f)
        return Path(f.name)


def create_test_default_config() -> dict:  # type: ignore[type-arg]
    """Create a test default config dictionary."""
    return {
        "name": "Test Default Config",
        "description": "Default configuration for testing",
        "version": "1.0.0",
        "benchmark_config": {
            "model": "test-model",
            "dataset_name": "test-dataset",
            "num_prompts": 100,
            "seed": 42,
        },
    }


def create_test_user_config() -> dict:  # type: ignore[type-arg]
    """Create a test user config dictionary."""
    return {
        "name": "Test User Config",
        "description": "User configuration for testing",
        "version": "1.1.0",
        "benchmark_config": {
            "model": "user-model",  # Override default
            "num_prompts": 200,  # Override default
        },
    }


class TestAddConfigFileArgToParser:
    """Test class for _add_config_file_arg_to_parser function."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality of _add_config_file_arg_to_parser."""
        parser = argparse.ArgumentParser()

        # Add the config file argument
        result_parser = _add_config_file_arg_to_parser(parser)

        # Verify the parser is returned (should be the same object)
        assert result_parser is parser

        # Test parsing with --config-file argument
        test_args = ["--config-file", "/path/to/config.yaml"]
        args = parser.parse_args(test_args)

        # Verify the argument was parsed correctly
        assert args.config_file == Path("/path/to/config.yaml")

    def test_without_config_file(self) -> None:
        """Test _add_config_file_arg_to_parser when --config-file is not provided."""
        parser = argparse.ArgumentParser()
        _add_config_file_arg_to_parser(parser)

        # Test parsing without --config-file argument (empty args)
        test_args: list[str] = []
        args = parser.parse_args(test_args)

        # Verify config_file is None when not provided
        assert args.config_file is None

    def test_help_text(self) -> None:
        """Test that the help text is correctly set for --config-file argument."""
        parser = argparse.ArgumentParser()
        _add_config_file_arg_to_parser(parser)

        # Get help text
        help_text = parser.format_help()

        # Verify help text contains expected content
        assert "--config-file" in help_text
        assert "Path to configuration file" in help_text
        assert "inherit from the default config" in help_text

    def test_type_conversion(self) -> None:
        """Test that --config-file argument correctly converts to Path type."""
        parser = argparse.ArgumentParser()
        _add_config_file_arg_to_parser(parser)

        # Test with different path formats
        test_cases = [
            "/absolute/path/config.yaml",
            "relative/path/config.yaml",
            "./config.yaml",
            "../config.yaml",
            "config.yaml",
        ]

        for path_str in test_cases:
            test_args = ["--config-file", path_str]
            args = parser.parse_args(test_args)

            # Verify the argument is converted to Path type
            assert isinstance(args.config_file, Path)
            # Use Path comparison instead of string comparison to handle normalization
            assert args.config_file == Path(path_str)


class TestResolveUserProvidedConfigFileCliArg:
    """Test class for _resolve_user_provided_config_file_cli_arg function."""

    def test_with_config_file(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg when --config-file is provided."""
        test_args = [
            "--config-file",
            "/path/to/config.yaml",
            "--other-arg",
            "value",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify remaining args exclude --config-file and its value
        assert remaining_args == ["--other-arg", "value"]

    def test_without_config_file(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg when --config-file is not provided."""
        test_args = ["--other-arg", "value", "--another-arg", "another-value"]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is None
        assert config_file_path is None

        # Verify all args remain unchanged
        assert remaining_args == test_args

    def test_empty_args(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with empty args list."""
        test_args: list[str] = []

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is None
        assert config_file_path is None

        # Verify remaining args is empty
        assert remaining_args == []

    def test_none_args(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with None args (uses sys.argv)."""
        # Mock sys.argv to simulate command line arguments
        test_argv = [
            "script.py",
            "--config-file",
            "/path/to/config.yaml",
            "--other-arg",
            "value",
        ]

        with patch.object(sys, "argv", test_argv):
            config_file_path, remaining_args = (
                _resolve_user_provided_config_file_cli_arg(args=None)
            )

            # Verify config file path is extracted correctly
            assert config_file_path == Path("/path/to/config.yaml")

            # Verify remaining args exclude script name, --config-file and its value
            assert remaining_args == ["--other-arg", "value"]

    def test_none_args_no_config(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with None args and no --config-file."""
        # Mock sys.argv to simulate command line arguments without --config-file
        test_argv = [
            "script.py",
            "--other-arg",
            "value",
            "--another-arg",
            "another-value",
        ]

        with patch.object(sys, "argv", test_argv):
            config_file_path, remaining_args = (
                _resolve_user_provided_config_file_cli_arg(args=None)
            )

            # Verify config file path is None
            assert config_file_path is None

            # Verify remaining args exclude script name
            assert remaining_args == [
                "--other-arg",
                "value",
                "--another-arg",
                "another-value",
            ]

    def test_multiple_config_files(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with multiple --config-file arguments."""
        # argparse.parse_known_args will parse the last occurrence of --config-file
        test_args = [
            "--config-file",
            "/path/to/first.yaml",
            "--other-arg",
            "value",
            "--config-file",
            "/path/to/second.yaml",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify the last config file path is extracted (argparse behavior)
        assert config_file_path == Path("/path/to/second.yaml")

        # Verify remaining args exclude both --config-file arguments
        assert remaining_args == ["--other-arg", "value"]

    def test_config_file_at_end(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with --config-file at the end."""
        test_args = [
            "--other-arg",
            "value",
            "--config-file",
            "/path/to/config.yaml",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify remaining args exclude --config-file and its value
        assert remaining_args == ["--other-arg", "value"]

    def test_invalid_config_file(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with invalid --config-file value."""
        # Test with --config-file but no value (should cause argparse error)
        test_args = ["--config-file", "--other-arg", "value"]

        with pytest.raises(SystemExit):
            # This should raise SystemExit due to missing argument value
            _resolve_user_provided_config_file_cli_arg(args=test_args)

    def test_unknown_args(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with unknown arguments."""
        test_args = [
            "--unknown-arg",
            "value",
            "--config-file",
            "/path/to/config.yaml",
            "--another-unknown",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify remaining args include unknown arguments
        assert remaining_args == ["--unknown-arg", "value", "--another-unknown"]

    def test_help_flag(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with --help flag."""
        test_args = ["--help", "--config-file", "/path/to/config.yaml"]

        # The preliminary parser has add_help=False, so --help won't cause SystemExit
        # Instead, it will be treated as an unknown argument and passed through
        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify --help is in remaining args
        assert remaining_args == ["--help"]

    def test_help_flag_only(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with only --help flag."""
        test_args = ["--help"]

        # The preliminary parser has add_help=False, so --help won't cause SystemExit
        # Instead, it will be treated as an unknown argument and passed through
        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is None
        assert config_file_path is None

        # Verify --help is in remaining args
        assert remaining_args == ["--help"]

    def test_verbose_flag(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with -v flag (should be ignored)."""
        test_args = [
            "-v",
            "--config-file",
            "/path/to/config.yaml",
            "--other-arg",
            "value",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify remaining args include -v and other args
        assert remaining_args == ["-v", "--other-arg", "value"]

    def test_positional_args(self) -> None:
        """Test _resolve_user_provided_config_file_cli_arg with positional arguments."""
        test_args = [
            "positional_arg",
            "--config-file",
            "/path/to/config.yaml",
            "another_positional",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify remaining args include positional arguments
        assert remaining_args == ["positional_arg", "another_positional"]

    def test_return_types(self) -> None:
        """Test that _resolve_user_provided_config_file_cli_arg returns correct types."""
        test_args = [
            "--config-file",
            "/path/to/config.yaml",
            "--other-arg",
            "value",
        ]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify return types
        assert isinstance(config_file_path, Path)
        assert isinstance(remaining_args, list)
        assert all(isinstance(arg, str) for arg in remaining_args)

    def test_no_config_file_return_types(self) -> None:
        """Test return types when no --config-file is provided."""
        test_args = ["--other-arg", "value"]

        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify return types
        assert config_file_path is None
        assert isinstance(remaining_args, list)
        assert all(isinstance(arg, str) for arg in remaining_args)


class TestLoadUserProvidedConfig:
    """Test class for _load_user_provided_config function."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality of _load_user_provided_config."""
        default_config = create_test_default_config()
        user_config = create_test_user_config()

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            # Load user config with inheritance
            result_config = _load_user_provided_config(
                user_path, default_path, BaseBenchmarkConfig
            )

            # Verify it's a BaseBenchmarkConfig instance
            assert isinstance(result_config, BaseBenchmarkConfig)

            # Verify inherited values from default config
            assert result_config.dataset_name == "test-dataset"
            assert result_config.seed == 42

            # Verify overridden values from user config
            assert result_config.model == "user-model"  # Overridden
            assert result_config.num_prompts == 200  # Overridden

        finally:
            # Clean up temporary files
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_empty_user_config(self) -> None:
        """Test _load_user_provided_config with empty user config."""
        default_config = create_test_default_config()
        user_config: dict = {  # type: ignore[type-arg]
            "benchmark_config": {}
        }  # Empty user config  # type: ignore[type-arg]

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            result_config = _load_user_provided_config(
                user_path, default_path, BaseBenchmarkConfig
            )

            # Should inherit all default values
            assert result_config.model == "test-model"
            assert result_config.dataset_name == "test-dataset"
            assert result_config.num_prompts == 100
            assert result_config.seed == 42

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_invalid_user_config_not_dict(self) -> None:
        """Test _load_user_provided_config with invalid user config (not a dict)."""
        default_config = create_test_default_config()
        # Create a YAML file with a list at the top level (not a dict)
        user_config = ["invalid", "config", "list"]

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            with pytest.raises(
                ValueError, match="must contain a dictionary at the top level"
            ):
                _load_user_provided_config(
                    user_path, default_path, BaseBenchmarkConfig
                )

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_invalid_default_config_not_dict(self) -> None:
        """Test _load_user_provided_config with invalid default config (not a dict)."""
        default_config = ["invalid", "config", "list"]
        user_config = create_test_user_config()

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            with pytest.raises(
                ValueError, match="must contain a dictionary at the top level"
            ):
                _load_user_provided_config(
                    user_path, default_path, BaseBenchmarkConfig
                )

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_nonexistent_user_config(self) -> None:
        """Test _load_user_provided_config with nonexistent user config file."""
        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        nonexistent_path = Path("/nonexistent/user_config.yaml")

        try:
            with pytest.raises(FileNotFoundError):
                _load_user_provided_config(
                    nonexistent_path, default_path, BaseBenchmarkConfig
                )

        finally:
            default_path.unlink(missing_ok=True)

    def test_nonexistent_default_config(self) -> None:
        """Test _load_user_provided_config with nonexistent default config file."""
        user_config = create_test_user_config()
        user_path = create_test_config_file(user_config)
        nonexistent_path = Path("/nonexistent/default_config.yaml")

        try:
            with pytest.raises(FileNotFoundError):
                _load_user_provided_config(
                    user_path, nonexistent_path, BaseBenchmarkConfig
                )

        finally:
            user_path.unlink(missing_ok=True)

    def test_invalid_yaml_user_config(self) -> None:
        """Test _load_user_provided_config with invalid YAML in user config file."""
        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)

        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            user_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                _load_user_provided_config(
                    user_path, default_path, BaseBenchmarkConfig
                )

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_invalid_yaml_default_config(self) -> None:
        """Test _load_user_provided_config with invalid YAML in default config file."""
        user_config = create_test_user_config()
        user_path = create_test_config_file(user_config)

        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            default_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                _load_user_provided_config(
                    user_path, default_path, BaseBenchmarkConfig
                )

        finally:
            user_path.unlink(missing_ok=True)
            default_path.unlink(missing_ok=True)

    def test_missing_benchmark_config_section(self) -> None:
        """Test _load_user_provided_config with user config missing benchmark_config section."""
        default_config = create_test_default_config()
        # Create user config without the benchmark_config section
        user_config = {
            "name": "Test User Config",
            "description": "User configuration without benchmark_config section",
            "version": "1.1.0",
            # Missing "benchmark_config" section
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            with patch(
                "max.benchmark.benchmark_shared.config.logger"
            ) as mock_logger:
                result_config = _load_user_provided_config(
                    user_path, default_path, BaseBenchmarkConfig
                )

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Cannot find benchmark_config section" in warning_call
            assert str(user_path) in warning_call
            assert (
                "Will not override benchmark config values from default config"
                in warning_call
            )

            # Verify it still loads with default values (no overrides)
            assert isinstance(result_config, BaseBenchmarkConfig)
            assert result_config.model == "test-model"  # From default
            assert result_config.dataset_name == "test-dataset"  # From default
            assert result_config.num_prompts == 100  # From default
            assert result_config.seed == 42  # From default

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_missing_benchmark_config_section_serving(self) -> None:
        """Test _load_user_provided_config with ServingBenchmarkConfig and missing section."""
        default_config = {
            "benchmark_config": {
                "model": "test-model",
                "dataset_name": "test-dataset",
                "num_prompts": 100,
                "seed": 42,
                "backend": "modular",
                "host": "localhost",
                "port": 8000,
            }
        }
        # Create user config without the benchmark_config section
        user_config = {
            "name": "Test User Config",
            "description": "User configuration without benchmark_config section",
            "version": "1.1.0",
            # Missing "benchmark_config" section
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            with patch(
                "max.benchmark.benchmark_shared.config.logger"
            ) as mock_logger:
                result_config = _load_user_provided_config(
                    user_path, default_path, ServingBenchmarkConfig
                )

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Cannot find benchmark_config section" in warning_call
            assert str(user_path) in warning_call
            assert (
                "Will not override benchmark config values from default config"
                in warning_call
            )

            # Verify it still loads with default values (no overrides)
            assert isinstance(result_config, ServingBenchmarkConfig)
            assert result_config.model == "test-model"  # From default
            assert result_config.dataset_name == "test-dataset"  # From default
            assert result_config.backend == "modular"  # From default
            assert result_config.host == "localhost"  # From default
            assert result_config.port == 8000  # From default

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_empty_benchmark_config_section(self) -> None:
        """Test _load_user_provided_config with empty benchmark_config section (should not warn)."""
        default_config = create_test_default_config()
        # Create user config with empty benchmark_config section
        user_config = {
            "name": "Test User Config",
            "description": "User configuration with empty benchmark_config section",
            "version": "1.1.0",
            "benchmark_config": {},  # Empty but present section
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            with patch(
                "max.benchmark.benchmark_shared.config.logger"
            ) as mock_logger:
                result_config = _load_user_provided_config(
                    user_path, default_path, BaseBenchmarkConfig
                )

            # Verify no warning was logged (section exists, even if empty)
            mock_logger.warning.assert_not_called()

            # Verify it loads with default values (empty section means no overrides)
            assert isinstance(result_config, BaseBenchmarkConfig)
            assert result_config.model == "test-model"  # From default
            assert result_config.dataset_name == "test-dataset"  # From default
            assert result_config.num_prompts == 100  # From default
            assert result_config.seed == 42  # From default

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)


class TestParseBenchmarkArgs:
    """Test class for parse_benchmark_args function with user-provided configs."""

    def test_with_user_config(self) -> None:
        """Test parse_benchmark_args with user-provided config file."""
        # Create test config files
        default_config = create_test_default_config()
        user_config = create_test_user_config()

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            # Test with user config file
            test_args = [
                "--config-file",
                str(user_path),
                "--model",
                "cli-override-model",
            ]

            with patch(
                "max.benchmark.benchmark_shared.config.logger"
            ) as mock_logger:
                args = parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test benchmark",
                    args=test_args,
                )

            # Verify the config file was used
            mock_logger.info.assert_any_call(
                f"Using user-provided configuration file: {user_path} (will inherit from {default_path})"
            )

            # Verify CLI overrides work
            assert args.model == "cli-override-model"

            # Verify inherited values
            assert args.dataset_name == "test-dataset"
            assert args.seed == 42

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_without_user_config(self) -> None:
        """Test parse_benchmark_args without user-provided config file."""
        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)

        try:
            # Test without config file
            test_args = ["--model", "cli-model"]

            with patch(
                "max.benchmark.benchmark_shared.config.logger"
            ) as mock_logger:
                args = parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test benchmark",
                    args=test_args,
                )

            # Verify default config was used
            mock_logger.info.assert_any_call(
                f"No configuration file path provided, using default {default_path} file"
            )

            # Verify CLI overrides work
            assert args.model == "cli-model"

            # Verify default values
            assert args.dataset_name == "test-dataset"
            assert args.seed == 42

        finally:
            default_path.unlink(missing_ok=True)

    def test_same_as_default_config(self) -> None:
        """Test parse_benchmark_args when user config is same as default config."""
        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)

        try:
            # Test with same file as default
            test_args = [
                "--config-file",
                str(default_path),
                "--model",
                "cli-model",
            ]

            with patch(
                "max.benchmark.benchmark_shared.config.logger"
            ) as mock_logger:
                args = parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test benchmark",
                    args=test_args,
                )

            # Verify it detected same file as default
            mock_logger.info.assert_any_call(
                f"Using default configuration file: {default_path}"
            )

            # Verify CLI overrides work
            assert args.model == "cli-model"

        finally:
            default_path.unlink(missing_ok=True)

    def test_with_serving_config(self) -> None:
        """Test parse_benchmark_args with ServingBenchmarkConfig."""
        # Create a serving-specific default config
        default_config = {
            "benchmark_config": {
                "model": "test-model",
                "dataset_name": "test-dataset",
                "num_prompts": 100,
                "seed": 42,
                "backend": "modular",
                "host": "localhost",
                "port": 8000,
            }
        }
        user_config = {
            "benchmark_config": {
                "backend": "vllm",  # Override default
                "dataset_name": "sharegpt",  # Override default
                "request_rate": "10.0",  # New serving-specific param
            }
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            test_args = [
                "--config-file",
                str(user_path),
                "--model",
                "test-model",
                "--dataset-name",
                "sharegpt",
            ]

            args = parse_benchmark_args(
                config_class=ServingBenchmarkConfig,
                default_config_path=default_path,
                description="Test serving benchmark",
                args=test_args,
            )

            # Verify it's a ServingBenchmarkConfig
            assert isinstance(args, argparse.Namespace)

            # Verify overridden values
            assert args.backend == "vllm"
            assert args.dataset_name == "sharegpt"

            # Verify new serving-specific values
            assert args.request_rate == "10.0"

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_invalid_user_config(self) -> None:
        """Test parse_benchmark_args with invalid user config file."""
        default_config = create_test_default_config()
        # Create invalid YAML content
        invalid_config = ["invalid", "yaml", "content"]

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(invalid_config)

        try:
            test_args = ["--config-file", str(user_path)]

            with pytest.raises((ValueError, yaml.YAMLError)):
                parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test benchmark",
                    args=test_args,
                )

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_nonexistent_user_config(self) -> None:
        """Test parse_benchmark_args with nonexistent user config file."""
        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        nonexistent_path = Path("/nonexistent/config.yaml")

        try:
            test_args = ["--config-file", str(nonexistent_path)]

            with pytest.raises(FileNotFoundError):
                parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test benchmark",
                    args=test_args,
                )

        finally:
            default_path.unlink(missing_ok=True)

    def test_nonexistent_default_config(self) -> None:
        """Test parse_benchmark_args with nonexistent default config file."""
        nonexistent_default_path = Path("/nonexistent/default.yaml")

        test_args = ["--model", "test-model"]

        with pytest.raises(FileNotFoundError):
            parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=nonexistent_default_path,
                description="Test benchmark",
                args=test_args,
            )

    def test_with_sys_argv(self) -> None:
        """Test parse_benchmark_args using sys.argv (args=None)."""
        default_config = create_test_default_config()
        user_config = create_test_user_config()

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            # Mock sys.argv
            test_argv = [
                "script.py",
                "--config-file",
                str(user_path),
                "--model",
                "cli-model",
            ]

            with patch.object(sys, "argv", test_argv):
                with patch(
                    "max.benchmark.benchmark_shared.config.logger"
                ) as mock_logger:
                    args = parse_benchmark_args(
                        config_class=BaseBenchmarkConfig,
                        default_config_path=default_path,
                        description="Test benchmark",
                        args=None,  # Use sys.argv
                    )

            # Verify user config was used
            mock_logger.info.assert_any_call(
                f"Using user-provided configuration file: {user_path} (will inherit from {default_path})"
            )

            # Verify CLI overrides work
            assert args.model == "cli-model"

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_complex_inheritance(self) -> None:
        """Test parse_benchmark_args with complex inheritance scenario."""
        # Create a more complex default config
        default_config = {
            "benchmark_config": {
                "model": "base-model",
                "dataset_name": "base-dataset",
                "num_prompts": 100,
                "seed": 42,
            }
        }

        # Create a user config that overrides some params
        user_config = {
            "benchmark_config": {
                "model": "user-model",  # Override
                "num_prompts": 200,  # Override
            }
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(user_config)

        try:
            test_args = [
                "--config-file",
                str(user_path),
                "--model",
                "cli-model",  # CLI override
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test complex inheritance",
                args=test_args,
            )

            # Verify CLI overrides take precedence
            assert args.model == "cli-model"

            # Verify user config overrides
            assert args.num_prompts == 200

            # Verify inherited values from default
            assert args.dataset_name == "base-dataset"
            assert args.seed == 42

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_empty_user_config(self) -> None:
        """Test parse_benchmark_args with empty user config file."""
        default_config = create_test_default_config()
        empty_user_config: dict = {"benchmark_config": {}}  # type: ignore[type-arg]

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(empty_user_config)

        try:
            test_args = [
                "--config-file",
                str(user_path),
                "--model",
                "cli-model",
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test empty user config",
                args=test_args,
            )

            # Should inherit all default values
            assert args.model == "cli-model"  # CLI override
            assert args.dataset_name == "test-dataset"  # From default
            assert args.seed == 42  # From default

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_partial_user_config(self) -> None:
        """Test parse_benchmark_args with partial user config (only some overrides)."""
        default_config = create_test_default_config()
        partial_user_config = {
            "benchmark_config": {
                "model": "partial-user-model",  # Override only this
                # Leave other values to inherit from default
            }
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(partial_user_config)

        try:
            test_args = [
                "--config-file",
                str(user_path),
                "--model",
                "cli-model",
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test partial user config",
                args=test_args,
            )

            # Verify CLI override takes precedence
            assert args.model == "cli-model"

            # Verify inherited values from default
            assert args.dataset_name == "test-dataset"
            assert args.seed == 42

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_complete_config_file_no_required_args(self) -> None:
        """Test that complete config file doesn't require CLI arguments for required fields."""
        # Create a complete config file with all required fields
        complete_config = {
            "benchmark_config": {
                "model": "google/gemma-3-27b-it",
                "dataset_name": "arxiv-summarization",
                "num_prompts": 50,
                "seed": 42,
            }
        }

        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(complete_config)

        try:
            # Test with complete config file - should not require any CLI args
            test_args = ["--config-file", str(user_path)]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test complete config",
                args=test_args,
            )

            # Verify all required fields are loaded from config
            assert args.model == "google/gemma-3-27b-it"
            assert args.dataset_name == "arxiv-summarization"
            assert args.num_prompts == 50
            assert args.seed == 42

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_incomplete_config_file_requires_missing_args(self) -> None:
        """Test that incomplete config file requires CLI arguments for missing required fields."""
        # Create an incomplete config file missing the model field
        # Note: dataset_name has a default value so it's never missing
        incomplete_config = {
            "benchmark_config": {
                "num_prompts": 50,
                "seed": 42,
                # Missing model (dataset_name has default value so not missing)
            }
        }

        # Create a default config with null values for required fields (like the real base config)
        default_config = {
            "name": "Test Default Config",
            "description": "Default configuration for testing",
            "version": "1.0.0",
            "benchmark_config": {
                "model": None,  # Required field set to None
                "dataset_name": None,  # This will be overridden by dataclass default
                "num_prompts": 100,
                "seed": 42,
            },
        }
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(incomplete_config)

        try:
            # Test with incomplete config file - should require missing CLI args
            test_args = ["--config-file", str(user_path)]

            with pytest.raises(SystemExit) as exc_info:
                parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test incomplete config",
                    args=test_args,
                )

            # Verify the error message mentions the missing required arguments
            # The exact error message format depends on argparse, but should mention required args
            assert (
                exc_info.value.code == 2
            )  # argparse exit code for missing required args

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_partial_config_file_requires_some_args(self) -> None:
        """Test that partial config file requires CLI arguments only for missing required fields."""
        # Create a partial config file with some required fields
        partial_config = {
            "benchmark_config": {
                "model": "google/gemma-3-27b-it",
                "num_prompts": 50,
                "seed": 42,
                # Missing dataset_name
            }
        }

        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(partial_config)

        try:
            # Test with partial config file - should require only missing CLI args
            test_args = [
                "--config-file",
                str(user_path),
                "--dataset-name",
                "arxiv-summarization",
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test partial config",
                args=test_args,
            )

            # Verify all fields are loaded (some from config, some from CLI)
            assert args.model == "google/gemma-3-27b-it"  # From config
            assert args.dataset_name == "arxiv-summarization"  # From CLI
            assert args.num_prompts == 50  # From config
            assert args.seed == 42  # From config

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_serving_config_smart_required_fields(self) -> None:
        """Test smart required fields logic with ServingBenchmarkConfig."""
        # Create a serving config with some required fields
        serving_config = {
            "benchmark_config": {
                "model": "google/gemma-3-27b-it",
                "backend": "modular",
                "host": "localhost",
                "port": 8000,
                # Missing dataset_name
            }
        }

        default_config = {
            "benchmark_config": {
                "model": "test-model",
                "dataset_name": "test-dataset",
                "num_prompts": 100,
                "seed": 42,
                "backend": "modular",
                "host": "localhost",
                "port": 8000,
            }
        }

        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(serving_config)

        try:
            # Test with serving config - should require only missing CLI args
            test_args = [
                "--config-file",
                str(user_path),
                "--dataset-name",
                "arxiv-summarization",
            ]

            args = parse_benchmark_args(
                config_class=ServingBenchmarkConfig,
                default_config_path=default_path,
                description="Test serving config",
                args=test_args,
            )

            # Verify all fields are loaded correctly
            assert args.model == "google/gemma-3-27b-it"  # From config
            assert args.dataset_name == "arxiv-summarization"  # From CLI
            assert args.backend == "modular"  # From config
            assert args.host == "localhost"  # From config
            assert args.port == 8000  # From config

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_none_values_in_config_require_cli_args(self) -> None:
        """Test that None values in config file require CLI arguments."""
        # Create a config file with None values for required fields
        none_config = {
            "benchmark_config": {
                "model": None,  # Explicitly None
                "dataset_name": None,  # Explicitly None
                "num_prompts": 50,
                "seed": 42,
            }
        }

        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(none_config)

        try:
            # Test with None values - should require CLI args
            test_args = ["--config-file", str(user_path)]

            with pytest.raises(SystemExit) as exc_info:
                parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test None values",
                    args=test_args,
                )

            # Verify the error message mentions the missing required arguments
            assert (
                exc_info.value.code == 2
            )  # argparse exit code for missing required args

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_empty_string_values_in_config_require_cli_args(self) -> None:
        """Test that empty string values in config file require CLI arguments."""
        # Create a config file with empty string values for required fields
        # Note: dataset_name has a default value so empty string gets overridden
        empty_config = {
            "benchmark_config": {
                "model": "",  # Empty string - should require CLI arg
                "dataset_name": "",  # Empty string - but has default value so not missing
                "num_prompts": 50,
                "seed": 42,
            }
        }

        # Create a default config with null values for required fields (like the real base config)
        default_config = {
            "name": "Test Default Config",
            "description": "Default configuration for testing",
            "version": "1.0.0",
            "benchmark_config": {
                "model": None,  # Required field set to None
                "dataset_name": None,  # This will be overridden by dataclass default
                "num_prompts": 100,
                "seed": 42,
            },
        }
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(empty_config)

        try:
            # Test with empty string values - should require CLI args
            test_args = ["--config-file", str(user_path)]

            with pytest.raises(SystemExit) as exc_info:
                parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test empty strings",
                    args=test_args,
                )

            # Verify the error message mentions the missing required arguments
            assert (
                exc_info.value.code == 2
            )  # argparse exit code for missing required args

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_cli_overrides_config_values(self) -> None:
        """Test that CLI arguments override config file values."""
        # Create a complete config file
        complete_config = {
            "benchmark_config": {
                "model": "config-model",
                "dataset_name": "sharegpt",  # Use valid dataset name
                "num_prompts": 50,
                "seed": 42,
            }
        }

        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(complete_config)

        try:
            # Test with CLI overrides - CLI should take precedence
            test_args = [
                "--config-file",
                str(user_path),
                "--model",
                "cli-model",
                "--dataset-name",
                "arxiv-summarization",  # Use valid dataset name
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test CLI overrides",
                args=test_args,
            )

            # Verify CLI overrides take precedence
            assert args.model == "cli-model"  # From CLI
            assert args.dataset_name == "arxiv-summarization"  # From CLI
            assert args.num_prompts == 50  # From config
            assert args.seed == 42  # From config

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_mixed_required_fields_scenario(self) -> None:
        """Test a complex scenario with mixed required and optional fields."""
        # Create a config with some required fields and some missing
        mixed_config = {
            "benchmark_config": {
                "model": "config-model",  # Provided in config
                "num_prompts": 200,  # Provided in config
                "seed": 99,  # Provided in config
                # Missing dataset_name
            }
        }

        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)
        user_path = create_test_config_file(mixed_config)

        try:
            # Test with mixed scenario - should require only missing CLI args
            test_args = [
                "--config-file",
                str(user_path),
                "--dataset-name",
                "arxiv-summarization",  # Use valid dataset name
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test mixed scenario",
                args=test_args,
            )

            # Verify the correct mix of config and CLI values
            assert args.model == "config-model"  # From config
            assert args.dataset_name == "arxiv-summarization"  # From CLI
            assert args.num_prompts == 200  # From config
            assert args.seed == 99  # From config

        finally:
            default_path.unlink(missing_ok=True)
            user_path.unlink(missing_ok=True)

    def test_no_config_file_requires_all_args(self) -> None:
        """Test that without config file, all required fields are required as CLI args."""
        # Create a default config with null values for required fields (like the real base config)
        # Note: dataset_name has a default value so it's never missing
        default_config = {
            "name": "Test Default Config",
            "description": "Default configuration for testing",
            "version": "1.0.0",
            "benchmark_config": {
                "model": None,  # Required field set to None
                "dataset_name": None,  # This will be overridden by dataclass default
                "num_prompts": 100,
                "seed": 42,
            },
        }
        default_path = create_test_config_file(default_config)

        try:
            # Test without config file - should require model CLI arg
            test_args: list[str] = []  # Missing model

            with pytest.raises(SystemExit) as exc_info:
                parse_benchmark_args(
                    config_class=BaseBenchmarkConfig,
                    default_config_path=default_path,
                    description="Test no config file",
                    args=test_args,
                )

            # Verify the error message mentions the missing required arguments
            assert (
                exc_info.value.code == 2
            )  # argparse exit code for missing required args

        finally:
            default_path.unlink(missing_ok=True)

    def test_no_config_file_with_all_args_works(self) -> None:
        """Test that without config file, providing all required CLI args works."""
        default_config = create_test_default_config()
        default_path = create_test_config_file(default_config)

        try:
            # Test without config file but with all required CLI args
            test_args = [
                "--model",
                "cli-model",
                "--dataset-name",
                "arxiv-summarization",  # Use valid dataset name
            ]

            args = parse_benchmark_args(
                config_class=BaseBenchmarkConfig,
                default_config_path=default_path,
                description="Test no config file with all args",
                args=test_args,
            )

            # Verify all values are loaded from CLI
            assert args.model == "cli-model"
            assert args.dataset_name == "arxiv-summarization"
            assert args.num_prompts == 100  # From default config
            assert args.seed == 42  # From default config

        finally:
            default_path.unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for multiple functions working together."""

    def test_add_config_file_arg_to_parser_integration_with_resolve(
        self,
    ) -> None:
        """Test integration between _add_config_file_arg_to_parser and _resolve_user_provided_config_file_cli_arg."""
        # Create a parser and add the config file argument
        parser = argparse.ArgumentParser()
        _add_config_file_arg_to_parser(parser)

        # Test with arguments that include --config-file
        test_args = ["--config-file", "/path/to/config.yaml"]

        # Use resolve function to extract config file
        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify the config file was extracted
        assert config_file_path == Path("/path/to/config.yaml")

        # Parse remaining args with the parser (should be empty)
        parsed_args = parser.parse_args(remaining_args)

        # Verify the parser can handle the remaining args (empty in this case)
        assert (
            parsed_args.config_file is None
        )  # Should be None since it was already parsed

    def test_help_integration(self) -> None:
        """Test that help integration works correctly with both functions."""
        parser = argparse.ArgumentParser()
        _add_config_file_arg_to_parser(parser)

        # Test help with --config-file in args
        test_args = ["--help", "--config-file", "/path/to/config.yaml"]

        # The preliminary parser has add_help=False, so --help won't cause SystemExit
        config_file_path, remaining_args = (
            _resolve_user_provided_config_file_cli_arg(args=test_args)
        )

        # Verify config file path is extracted correctly
        assert config_file_path == Path("/path/to/config.yaml")

        # Verify --help is in remaining args
        assert remaining_args == ["--help"]


class TestSweepServingBenchmarkConfig:
    """Test class for SweepServingBenchmarkConfig."""

    def test_sweep_config_creation(self) -> None:
        """Test that SweepServingBenchmarkConfig can be created with default values."""
        config = SweepServingBenchmarkConfig()

        # Test that all sweep-specific fields have correct default values
        assert config.workload_config == ""
        assert config.log_dir is None
        assert config.dry_run is False
        assert config.upload_results is False
        assert config.benchmark_sha is None
        assert config.cluster_information_path is None
        assert config.benchmark_config_name is None
        assert config.metadata == []
        assert config.latency_percentiles == "50,90,95,99"
        assert config.num_iters == 1
        assert config.max_concurrency is None
        assert config.request_rate == "inf"

    def test_sweep_config_inheritance(self) -> None:
        """Test that SweepServingBenchmarkConfig inherits from ServingBenchmarkConfig."""
        config = SweepServingBenchmarkConfig()

        # Test that it inherits serving-specific fields
        assert hasattr(config, "backend")
        assert hasattr(config, "host")
        assert hasattr(config, "port")
        assert hasattr(config, "endpoint")

        # Test that it inherits base fields
        assert hasattr(config, "model")
        assert hasattr(config, "dataset_name")
        assert hasattr(config, "num_prompts")
        assert hasattr(config, "seed")

    def test_sweep_config_required_fields(self) -> None:
        """Test that SweepServingBenchmarkConfig has correct required fields."""
        required_fields = (
            SweepServingBenchmarkConfig.get_default_required_fields()
        )

        # Should include workload_config as required
        assert "workload_config" in required_fields

        # Should include inherited required fields
        assert "model" in required_fields
        # dataset_name is not required for SweepServingBenchmarkConfig
        # (it's parsed from workload config instead)

    def test_sweep_config_field_metadata(self) -> None:
        """Test that SweepServingBenchmarkConfig fields have proper metadata."""
        config = SweepServingBenchmarkConfig()

        # Test that fields have proper group metadata
        from dataclasses import fields

        for field_info in fields(config):
            if field_info.name in ["workload_config", "log_dir", "dry_run"]:
                assert "group" in field_info.metadata
                assert field_info.metadata["group"] in [
                    "Workload Configuration",
                    "Logging and Debugging",
                ]
            elif field_info.name in [
                "upload_results",
                "benchmark_sha",
                "cluster_information_path",
                "benchmark_config_name",
            ]:
                assert "group" in field_info.metadata
                assert (
                    field_info.metadata["group"]
                    == "Result Upload Configuration"
                )
            elif field_info.name in ["metadata", "latency_percentiles"]:
                assert "group" in field_info.metadata
                assert (
                    field_info.metadata["group"]
                    == "Metadata and Result Tracking"
                )
            elif field_info.name in [
                "num_iters",
            ]:
                assert "group" in field_info.metadata
                assert field_info.metadata["group"] == "Sweep Configuration"
            elif field_info.name in [
                "max_concurrency",
            ]:
                assert "group" in field_info.metadata
                assert field_info.metadata["group"] == "Request Configuration"
            elif field_info.name in [
                "request_rate",
            ]:
                assert "group" in field_info.metadata
                assert field_info.metadata["group"] == "Traffic Control"


# ===----------------------------------------------------------------------=== #
# Sweepable Type Tests
# ===----------------------------------------------------------------------=== #


@dataclass
class TestSweepableConfig(BaseBenchmarkConfig):
    """Test configuration class with sweepable_type fields for testing."""

    # Integer sweepable field
    max_concurrency: str | None = field(
        default=None,
        metadata={
            "group": "Request Configuration",
            "group_description": "Parameters controlling request concurrency and processing",
            "sweepable_type": int,
        },
    )
    """Maximum concurrent requests. Can be a single integer, "None", or comma-separated string for sweep configs."""

    # Float sweepable field
    request_rate: str | None = field(
        default="inf",
        metadata={
            "group": "Traffic Control",
            "group_description": "Parameters controlling request rate and traffic patterns",
            "sweepable_type": float,
        },
    )
    """Requests per second. Can be a single float value or comma-separated string for sweep configs."""

    # Non-sweepable field for comparison
    model: str = field(
        default="test-model",
        metadata={
            "group": "Model Configuration",
            "group_description": "Model configuration parameters",
        },
    )
    """Model name (not sweepable)."""


def parse_sweepable_values(
    value: str | None, sweepable_type: type
) -> list[int | float | None]:
    """
    Parse comma-separated values according to sweepable_type.

    This function mimics the parsing logic used in sweep-benchmark-serving.py
    for handling sweepable_type fields.

    Args:
        value: Comma-separated string of values to parse
        sweepable_type: Type to parse values as (int or float)

    Returns:
        List of parsed values

    Raises:
        ValueError: If value cannot be parsed as the specified type
    """
    if not value or value.lower() == "none":
        return []

    # Handle "inf" for float types
    if sweepable_type is float and value.lower() == "inf":
        return [float("inf")]

    try:
        # Split by comma and parse each value
        values = [x.strip() for x in value.split(",")]
        parsed_values: list[int | float | None] = []

        for val in values:
            if not val:  # Handle empty strings
                parsed_values.append(None)
            elif val.lower() == "none":
                parsed_values.append(None)
            elif val.lower() == "inf" and sweepable_type is float:
                parsed_values.append(float("inf"))
            else:
                # Handle float values that can be converted to int
                if sweepable_type is int and "." in val:
                    # Convert float string to int (truncates decimal part)
                    parsed_values.append(int(float(val)))
                else:
                    parsed_values.append(sweepable_type(val))

        return parsed_values
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Cannot parse '{value}' as {sweepable_type.__name__}: {e}"
        ) from e


class TestSweepableTypeParsing:
    """Test class for sweepable_type parsing functionality."""

    def test_parse_int_values_basic(self) -> None:
        """Test basic integer parsing for sweepable_type: int fields."""
        # Test single integer
        result = parse_sweepable_values("5", int)
        assert result == [5]

        # Test comma-separated integers
        result = parse_sweepable_values("1,2,3,4,5", int)
        assert result == [1, 2, 3, 4, 5]

        # Test with spaces
        result = parse_sweepable_values("1, 2, 3, 4, 5", int)
        assert result == [1, 2, 3, 4, 5]

    def test_parse_int_values_with_none(self) -> None:
        """Test integer parsing with None values."""
        # Test "None" string
        result = parse_sweepable_values("None", int)
        assert result == []

        # Test mixed with None
        result = parse_sweepable_values("1,None,3", int)
        assert result == [1, None, 3]

        # Test all None
        result = parse_sweepable_values("None,None,None", int)
        assert result == [None, None, None]

    def test_parse_float_values_basic(self) -> None:
        """Test basic float parsing for sweepable_type: float fields."""
        # Test single float
        result = parse_sweepable_values("1.5", float)
        assert result == [1.5]

        # Test comma-separated floats
        result = parse_sweepable_values("1.0,2.5,3.14,4.0", float)
        assert result == [1.0, 2.5, 3.14, 4.0]

        # Test with spaces
        result = parse_sweepable_values("1.0, 2.5, 3.14, 4.0", float)
        assert result == [1.0, 2.5, 3.14, 4.0]

    def test_parse_float_values_with_inf(self) -> None:
        """Test float parsing with infinity values."""
        # Test "inf" string
        result = parse_sweepable_values("inf", float)
        assert result == [float("inf")]

        # Test mixed with inf
        result = parse_sweepable_values("1.0,inf,3.0", float)
        assert result == [1.0, float("inf"), 3.0]

        # Test all inf
        result = parse_sweepable_values("inf,inf,inf", float)
        assert result == [float("inf"), float("inf"), float("inf")]

    def test_parse_float_values_with_none(self) -> None:
        """Test float parsing with None values."""
        # Test "None" string
        result = parse_sweepable_values("None", float)
        assert result == []

        # Test mixed with None
        result = parse_sweepable_values("1.0,None,3.0", float)
        assert result == [1.0, None, 3.0]

    def test_parse_empty_values(self) -> None:
        """Test parsing empty values."""
        # Test empty string
        result = parse_sweepable_values("", int)
        assert result == []

        result = parse_sweepable_values("", float)
        assert result == []

    def test_parse_single_value(self) -> None:
        """Test parsing single values without commas."""
        # Test single integer
        result = parse_sweepable_values("42", int)
        assert result == [42]

        # Test single float
        result = parse_sweepable_values("3.14", float)
        assert result == [3.14]

    def test_parse_whitespace_handling(self) -> None:
        """Test that whitespace is properly handled."""
        # Test leading/trailing spaces
        result = parse_sweepable_values(" 1 , 2 , 3 ", int)
        assert result == [1, 2, 3]

        # Test multiple spaces
        result = parse_sweepable_values("1,  2,   3", int)
        assert result == [1, 2, 3]

    def test_parse_invalid_values(self) -> None:
        """Test parsing invalid values raises appropriate errors."""
        # Test invalid integer
        with pytest.raises(ValueError, match="Cannot parse 'abc' as int"):
            parse_sweepable_values("abc", int)

        # Test invalid float
        with pytest.raises(ValueError, match="Cannot parse 'xyz' as float"):
            parse_sweepable_values("xyz", float)

        # Test mixed valid/invalid
        with pytest.raises(ValueError, match="Cannot parse '1,abc,3' as int"):
            parse_sweepable_values("1,abc,3", int)

    def test_parse_edge_cases(self) -> None:
        """Test edge cases in parsing."""
        # Test zero values
        result = parse_sweepable_values("0,0.0", int)
        assert result == [0, 0]

        result = parse_sweepable_values("0,0.0", float)
        assert result == [0.0, 0.0]

        # Test negative values
        result = parse_sweepable_values("-1,-2.5", int)
        assert result == [-1, -2]

        result = parse_sweepable_values("-1,-2.5", float)
        assert result == [-1.0, -2.5]

        # Test scientific notation
        result = parse_sweepable_values("1e2,2.5e-1", float)
        assert result == [100.0, 0.25]

    def test_parse_large_values(self) -> None:
        """Test parsing large values."""
        # Test large integers
        result = parse_sweepable_values("1000000,2000000", int)
        assert result == [1000000, 2000000]

        # Test large floats
        result = parse_sweepable_values("1e6,2e6", float)
        assert result == [1000000.0, 2000000.0]

    def test_parse_mixed_types_in_string(self) -> None:
        """Test that string contains mixed types that can be parsed."""
        # Test integers that can be parsed as floats
        result = parse_sweepable_values("1,2,3", float)
        assert result == [1.0, 2.0, 3.0]

        # Test floats that can be parsed as integers (truncated)
        result = parse_sweepable_values("1.0,2.0,3.0", int)
        assert result == [1, 2, 3]

    def test_parse_case_insensitive_none_inf(self) -> None:
        """Test that None and inf parsing is case insensitive."""
        # Test case insensitive None
        result = parse_sweepable_values("NONE", int)
        assert result == []

        result = parse_sweepable_values("none", int)
        assert result == []

        result = parse_sweepable_values("None", int)
        assert result == []

        # Test case insensitive inf
        result = parse_sweepable_values("INF", float)
        assert result == [float("inf")]

        result = parse_sweepable_values("inf", float)
        assert result == [float("inf")]

        result = parse_sweepable_values("Inf", float)
        assert result == [float("inf")]

    def test_parse_complex_combinations(self) -> None:
        """Test complex combinations of values."""
        # Test complex integer combination
        result = parse_sweepable_values("1,None,3,None,5", int)
        assert result == [1, None, 3, None, 5]

        # Test complex float combination
        result = parse_sweepable_values("1.0,inf,3.14,None,5.0", float)
        assert result == [1.0, float("inf"), 3.14, None, 5.0]

    def test_parse_duplicate_values(self) -> None:
        """Test parsing duplicate values."""
        # Test duplicate integers
        result = parse_sweepable_values("1,1,1", int)
        assert result == [1, 1, 1]

        # Test duplicate floats
        result = parse_sweepable_values("1.0,1.0,1.0", float)
        assert result == [1.0, 1.0, 1.0]

    def test_parse_single_comma(self) -> None:
        """Test parsing single comma (edge case)."""
        # Test single comma
        result = parse_sweepable_values(",", int)
        assert result == [None, None]  # Empty strings become None

        result = parse_sweepable_values(",", float)
        assert result == [None, None]

    def test_parse_trailing_comma(self) -> None:
        """Test parsing with trailing comma."""
        # Test trailing comma
        result = parse_sweepable_values("1,2,", int)
        assert result == [1, 2, None]  # Empty string becomes None

        result = parse_sweepable_values("1.0,2.0,", float)
        assert result == [1.0, 2.0, None]

    def test_parse_leading_comma(self) -> None:
        """Test parsing with leading comma."""
        # Test leading comma
        result = parse_sweepable_values(",1,2", int)
        assert result == [None, 1, 2]  # Empty string becomes None

        result = parse_sweepable_values(",1.0,2.0", float)
        assert result == [None, 1.0, 2.0]


class TestSweepableTypeIntegration:
    """Integration tests for sweepable_type with actual config classes."""

    def test_config_field_metadata(self) -> None:
        """Test that config fields have correct sweepable_type metadata."""
        config = TestSweepableConfig()

        # Get field metadata
        from dataclasses import fields

        field_dict = {field.name: field for field in fields(config)}

        # Test max_concurrency field
        max_concurrency_field = field_dict["max_concurrency"]
        assert "sweepable_type" in max_concurrency_field.metadata
        assert max_concurrency_field.metadata["sweepable_type"] is int

        # Test request_rate field
        request_rate_field = field_dict["request_rate"]
        assert "sweepable_type" in request_rate_field.metadata
        assert request_rate_field.metadata["sweepable_type"] is float

        # Test model field (not sweepable)
        model_field = field_dict["model"]
        assert "sweepable_type" not in model_field.metadata

    def test_sweepable_type_parsing_integration(self) -> None:
        """Test integration of sweepable_type parsing with config values."""
        # Test with max_concurrency (int type)
        config = TestSweepableConfig(max_concurrency="1,2,3")

        # Simulate the parsing that would happen in sweep script
        if config.max_concurrency:
            field_metadata = fields(TestSweepableConfig)[
                0
            ].metadata  # max_concurrency field
            sweepable_type = field_metadata.get("sweepable_type")
            if sweepable_type:
                parsed_values = parse_sweepable_values(
                    config.max_concurrency, sweepable_type
                )
                assert parsed_values == [1, 2, 3]

        # Test with request_rate (float type)
        config = TestSweepableConfig(request_rate="1.0,2.5,inf")

        if config.request_rate:
            field_metadata = fields(TestSweepableConfig)[
                1
            ].metadata  # request_rate field
            sweepable_type = field_metadata.get("sweepable_type")
            if sweepable_type:
                parsed_values = parse_sweepable_values(
                    config.request_rate, sweepable_type
                )
                assert parsed_values == [1.0, 2.5, float("inf")]

    def test_sweepable_type_with_none_values(self) -> None:
        """Test sweepable_type parsing with None values in config."""
        # Test with None max_concurrency
        config = TestSweepableConfig(max_concurrency=None)

        if config.max_concurrency:
            field_metadata = fields(TestSweepableConfig)[0].metadata
            sweepable_type = field_metadata.get("sweepable_type")
            if sweepable_type:
                parsed_values = parse_sweepable_values(
                    config.max_concurrency, sweepable_type
                )
                assert parsed_values == []
        else:
            # None values should result in empty list
            assert True  # This is expected behavior

    def test_sweepable_type_with_default_values(self) -> None:
        """Test sweepable_type parsing with default values."""
        # Test with default request_rate ("inf")
        config = TestSweepableConfig()

        field_metadata = fields(TestSweepableConfig)[
            1
        ].metadata  # request_rate field
        sweepable_type = field_metadata.get("sweepable_type")
        if sweepable_type:
            parsed_values = parse_sweepable_values(
                config.request_rate, sweepable_type
            )
            assert parsed_values == [float("inf")]

    def test_sweepable_type_error_handling(self) -> None:
        """Test error handling in sweepable_type parsing."""
        # Test with invalid max_concurrency
        config = TestSweepableConfig(max_concurrency="1,abc,3")

        field_metadata = fields(TestSweepableConfig)[0].metadata
        sweepable_type = field_metadata.get("sweepable_type")
        if sweepable_type:
            with pytest.raises(
                ValueError, match="Cannot parse '1,abc,3' as int"
            ):
                parse_sweepable_values(config.max_concurrency, sweepable_type)

    def test_sweepable_type_with_real_serving_config(self) -> None:
        """Test sweepable_type parsing with actual ServingBenchmarkConfig fields."""
        # Test with actual serving config
        config = ServingBenchmarkConfig(
            max_concurrency="1,2,4,8", request_rate="1.0,2.0,4.0,8.0"
        )

        # Get field metadata
        from dataclasses import fields

        field_dict = {field.name: field for field in fields(config)}

        # Test max_concurrency parsing
        max_concurrency_field = field_dict["max_concurrency"]
        sweepable_type = max_concurrency_field.metadata.get("sweepable_type")
        if sweepable_type and config.max_concurrency:
            parsed_values = parse_sweepable_values(
                config.max_concurrency, sweepable_type
            )
            assert parsed_values == [1, 2, 4, 8]

        # Test request_rate parsing
        request_rate_field = field_dict["request_rate"]
        sweepable_type = request_rate_field.metadata.get("sweepable_type")
        if sweepable_type and config.request_rate:
            parsed_values = parse_sweepable_values(
                config.request_rate, sweepable_type
            )
            assert parsed_values == [1.0, 2.0, 4.0, 8.0]


class TestSweepableTypeValidation:
    """Test validation and error handling for sweepable_type functionality."""

    def test_validate_sweepable_type_metadata(self) -> None:
        """Test that sweepable_type metadata is properly validated."""
        from dataclasses import fields

        # Test that only int and float are supported
        config = TestSweepableConfig()
        field_dict = {field.name: field for field in fields(config)}

        # Check max_concurrency field
        max_concurrency_field = field_dict["max_concurrency"]
        sweepable_type = max_concurrency_field.metadata.get("sweepable_type")
        assert sweepable_type in [int, float]

        # Check request_rate field
        request_rate_field = field_dict["request_rate"]
        sweepable_type = request_rate_field.metadata.get("sweepable_type")
        assert sweepable_type in [int, float]

    def test_validate_parsed_values_types(self) -> None:
        """Test that parsed values have correct types."""
        # Test integer parsing returns integers
        result = parse_sweepable_values("1,2,3", int)
        assert all(isinstance(x, int) or x is None for x in result)

        # Test float parsing returns floats
        result = parse_sweepable_values("1.0,2.0,3.0", float)
        assert all(isinstance(x, float) or x is None for x in result)

    def test_validate_error_messages(self) -> None:
        """Test that error messages are informative."""
        # Test integer error message
        with pytest.raises(ValueError) as exc_info:
            parse_sweepable_values("abc", int)
        assert "Cannot parse 'abc' as int" in str(exc_info.value)

        # Test float error message
        with pytest.raises(ValueError) as exc_info:
            parse_sweepable_values("xyz", float)
        assert "Cannot parse 'xyz' as float" in str(exc_info.value)

    def test_validate_edge_case_handling(self) -> None:
        """Test that edge cases are handled correctly."""
        # Test empty string
        result = parse_sweepable_values("", int)
        assert result == []

        # Test single comma
        result = parse_sweepable_values(",", int)
        assert result == [None, None]

        # Test multiple commas
        result = parse_sweepable_values(",,,", int)
        assert result == [None, None, None, None]

    def test_validate_infinity_handling(self) -> None:
        """Test that infinity is handled correctly for float types."""
        # Test inf for float
        result = parse_sweepable_values("inf", float)
        assert result == [float("inf")]
        assert all(x == float("inf") for x in result)

        # Test inf should not work for int
        with pytest.raises(ValueError):
            parse_sweepable_values("inf", int)

    def test_validate_none_handling(self) -> None:
        """Test that None is handled correctly."""
        # Test None for both types
        result_int = parse_sweepable_values("None", int)
        result_float = parse_sweepable_values("None", float)

        assert result_int == []
        assert result_float == []

        # Test mixed None values
        result = parse_sweepable_values("1,None,3", int)
        assert result == [1, None, 3]
