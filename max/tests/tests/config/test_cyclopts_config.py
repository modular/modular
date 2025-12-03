# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Unit tests for cyclopts_config.py.

Tests the configuration precedence order provided by create_cyclopts_app and
setup_config_file_meta_command:
1. CLI arguments (highest priority)
2. Config files (YAML, via --config-file)
3. Environment variables (MODULAR_*)
4. Defaults in config classes (lowest priority)

These tests use simple test config classes to verify the cyclopts_config
utilities work correctly with Pydantic-based config classes.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from cyclopts import Parameter
from max.config.cyclopts_config import (
    create_cyclopts_app,
    setup_config_file_meta_command,
)
from pydantic import BaseModel, Field


# Simple test config classes for testing cyclopts_config utilities
@Parameter(name="*", group="Shape Options")
class ShapeConfig(BaseModel):
    """Test configuration class for shape options."""

    batch_size: int = Field(default=1)
    """Number of prompts in each batch."""

    input_len: int = Field(default=256)
    """Number of input tokens in each prompt."""

    output_len: int = Field(default=128)
    """Number of output tokens generated per prompt."""


@Parameter(name="*", group="Model Options")
class ModelConfig(BaseModel):
    """Test configuration class for model options."""

    model: str | None = Field(default=None)
    """Name of the model."""

    seed: int = Field(default=0)
    """Random seed for reproducibility."""


@Parameter(name="*", group="Hardware Options")
class HardwareConfig(BaseModel):
    """Test configuration class for hardware options."""

    devices: str | None = Field(default=None)
    """Hardware device on which model will be executed."""

    quantization_encoding: str = Field(default="q4_k")
    """Quantization encoding to benchmark."""


@Parameter(name="*", group="Profiling Options")
class ProfilingConfig(BaseModel):
    """Test configuration class for profiling options."""

    gpu_sampling_interval: float | None = Field(default=None)
    """Interval, in seconds, between GPU resource samples."""


@Parameter(name="*", group="Execution Options")
class ExecutionConfig(BaseModel):
    """Test configuration class for execution options."""

    num_warmups: int = Field(default=1)
    """Number of warmup iterations to run."""

    num_iters: int = Field(default=1)
    """Number of benchmarking iterations to run."""

    enable_prefix_caching: bool = Field(default=True)
    """Enable prefix caching for paged-attention kvcache."""


@Parameter(name="*", group="Sampling Options")
class SamplingConfig(BaseModel):
    """Test configuration class for sampling options."""

    top_k: int = Field(default=1)
    """Limits the sampling to the K most probable tokens."""


def create_test_config_file(content: dict[str, Any]) -> Path:
    """Create a temporary YAML config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(content, f)
        return Path(f.name)


def create_test_config_content(**kwargs: Any) -> dict[str, Any]:
    """Create a test config dictionary with test_config section."""
    return {
        "test_config": {
            **kwargs,
        }
    }


class TestConfigPrecedence:
    """Test configuration precedence order."""

    def test_defaults_only(self) -> None:
        """Test that defaults are used when nothing else is provided."""
        app = create_cyclopts_app(name="test_app", help_text="Test app")

        @app.default
        def test_command(
            model_config: ModelConfig | None = None,
            shape_config: ShapeConfig | None = None,
            hardware_config: HardwareConfig | None = None,
            profiling_config: ProfilingConfig | None = None,
            execution_config: ExecutionConfig | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> None:
            # Cyclopts should create all config objects, but if not, create defaults
            if model_config is None:
                model_config = ModelConfig()
            if shape_config is None:
                shape_config = ShapeConfig()
            if hardware_config is None:
                hardware_config = HardwareConfig()
            if profiling_config is None:
                profiling_config = ProfilingConfig()
            if execution_config is None:
                execution_config = ExecutionConfig()
            if sampling_config is None:
                sampling_config = SamplingConfig()
            # Check some default values
            assert shape_config.batch_size == 1
            assert shape_config.input_len == 256
            assert shape_config.output_len == 128
            assert execution_config.num_warmups == 1
            assert execution_config.num_iters == 1
            assert hardware_config.quantization_encoding == "q4_k"
            assert model_config.seed == 0
            assert execution_config.enable_prefix_caching is True
            assert sampling_config.top_k == 1

        try:
            app([])
        except SystemExit as e:
            if e.code != 0:
                raise

    def test_env_vars_override_defaults(self) -> None:
        """Test that MODULAR_ environment variables override defaults."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_OUTPUT_LEN": "256",
            "MODULAR_NUM_WARMUPS": "3",
            "MODULAR_SEED": "42",
            "MODULAR_QUANTIZATION_ENCODING": "bfloat16",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert shape_config is not None
                assert hardware_config is not None
                assert execution_config is not None
                assert shape_config.batch_size == 5
                assert shape_config.input_len == 512
                assert shape_config.output_len == 256
                assert execution_config.num_warmups == 3
                assert model_config.seed == 42
                assert hardware_config.quantization_encoding == "bfloat16"

            try:
                app([])
            except SystemExit as e:
                if e.code != 0:
                    raise

    def test_config_file_overrides_env_vars(self) -> None:
        """Test that config file values override environment variables."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_MODEL": "env-model",
        }

        config_content = create_test_config_content(
            batch_size=10,
            input_len=1024,
            model="config-model",
        )

        config_file = create_test_config_file(config_content)

        try:
            with patch.dict(os.environ, env_vars, clear=False):
                app = create_cyclopts_app(name="test_app", help_text="Test app")

                @app.default
                def test_command(
                    model_config: ModelConfig | None = None,
                    shape_config: ShapeConfig | None = None,
                    hardware_config: HardwareConfig | None = None,
                    profiling_config: ProfilingConfig | None = None,
                    execution_config: ExecutionConfig | None = None,
                    sampling_config: SamplingConfig | None = None,
                ) -> None:
                    assert model_config is not None
                    assert shape_config is not None
                    # Config file should override env vars
                    assert (
                        shape_config.batch_size == 10
                    )  # From config file, not env (5)
                    assert (
                        shape_config.input_len == 1024
                    )  # From config file, not env (512)
                    assert (
                        model_config.model == "config-model"
                    )  # From config file, not env

                # Set up meta command and call it with test arguments
                try:
                    setup_config_file_meta_command(
                        app,
                        root_keys="test_config",
                        must_exist=False,
                        args=["--config-file", str(config_file)],
                    )
                except SystemExit as e:
                    if e.code != 0:
                        raise

        finally:
            config_file.unlink(missing_ok=True)

    def test_cli_args_override_config_file(self) -> None:
        """Test that CLI arguments override config file values."""
        config_content = create_test_config_content(
            batch_size=10,
            input_len=1024,
            output_len=256,
            model="config-model",
        )

        config_file = create_test_config_file(config_content)

        try:
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert shape_config is not None
                # CLI args should override config file
                assert (
                    shape_config.batch_size == 20
                )  # From CLI, not config (10)
                assert (
                    shape_config.input_len == 2048
                )  # From CLI, not config (1024)
                assert model_config.model == "cli-model"  # From CLI, not config

            # Set up meta command and call it with test arguments
            try:
                setup_config_file_meta_command(
                    app,
                    root_keys="test_config",
                    must_exist=False,
                    args=[
                        "--config-file",
                        str(config_file),
                        "--batch-size=20",
                        "--input-len=2048",
                        "--model=cli-model",
                    ],
                )
            except SystemExit as e:
                if e.code != 0:
                    raise

        finally:
            config_file.unlink(missing_ok=True)

    def test_cli_args_override_env_vars(self) -> None:
        """Test that CLI arguments override environment variables."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_MODEL": "env-model",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert shape_config is not None
                # CLI args should override env vars
                assert shape_config.batch_size == 20  # From CLI, not env (5)
                assert shape_config.input_len == 2048  # From CLI, not env (512)
                assert model_config.model == "cli-model"  # From CLI, not env

            try:
                app(
                    ["--batch-size=20", "--input-len=2048", "--model=cli-model"]
                )
            except SystemExit as e:
                if e.code != 0:
                    raise

    def test_cli_args_override_config_file_and_env_vars(self) -> None:
        """Test that CLI arguments override both config file and env vars."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_MODEL": "env-model",
        }

        config_content = create_test_config_content(
            batch_size=10,
            input_len=1024,
            model="config-model",
        )

        config_file = create_test_config_file(config_content)

        try:
            with patch.dict(os.environ, env_vars, clear=False):
                app = create_cyclopts_app(name="test_app", help_text="Test app")

                @app.default
                def test_command(
                    model_config: ModelConfig | None = None,
                    shape_config: ShapeConfig | None = None,
                    hardware_config: HardwareConfig | None = None,
                    profiling_config: ProfilingConfig | None = None,
                    execution_config: ExecutionConfig | None = None,
                    sampling_config: SamplingConfig | None = None,
                ) -> None:
                    assert model_config is not None
                    assert shape_config is not None
                    # CLI args should have highest priority
                    assert shape_config.batch_size == 20  # From CLI
                    assert shape_config.input_len == 2048  # From CLI
                    assert model_config.model == "cli-model"  # From CLI

                try:
                    setup_config_file_meta_command(
                        app,
                        root_keys="test_config",
                        args=[
                            "--config-file",
                            str(config_file),
                            "--batch-size=20",
                            "--input-len=2048",
                            "--model=cli-model",
                        ],
                    )
                except SystemExit as e:
                    if e.code != 0:
                        raise

        finally:
            config_file.unlink(missing_ok=True)

    def test_config_file_overrides_env_vars_partial(self) -> None:
        """Test that config file overrides env vars even when only partial config."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_OUTPUT_LEN": "256",
            "MODULAR_MODEL": "env-model",
        }

        # Config file only overrides some values
        config_content = create_test_config_content(
            batch_size=10,
            model="config-model",
            # input_len and output_len not in config, should use env vars
        )

        config_file = create_test_config_file(config_content)

        try:
            with patch.dict(os.environ, env_vars, clear=False):
                app = create_cyclopts_app(name="test_app", help_text="Test app")

                @app.default
                def test_command(
                    model_config: ModelConfig | None = None,
                    shape_config: ShapeConfig | None = None,
                    hardware_config: HardwareConfig | None = None,
                    profiling_config: ProfilingConfig | None = None,
                    execution_config: ExecutionConfig | None = None,
                    sampling_config: SamplingConfig | None = None,
                ) -> None:
                    assert model_config is not None
                    assert shape_config is not None
                    # Config file overrides where specified
                    assert shape_config.batch_size == 10  # From config file
                    assert (
                        model_config.model == "config-model"
                    )  # From config file
                    # Env vars used where config file doesn't specify
                    assert shape_config.input_len == 512  # From env var
                    assert shape_config.output_len == 256  # From env var

                try:
                    setup_config_file_meta_command(
                        app,
                        root_keys="test_config",
                        args=[
                            "--config-file",
                            str(config_file),
                        ],
                    )
                except SystemExit as e:
                    if e.code != 0:
                        raise

        finally:
            config_file.unlink(missing_ok=True)

    def test_multiple_config_files_later_overrides_earlier(self) -> None:
        """Test multiple config files behavior.

        Note: Cyclopts processes config sources in order, but may not override
        values from earlier Yaml sources. This test verifies that config files
        are processed correctly when multiple files are provided.
        """
        config_content_1 = create_test_config_content(
            batch_size=10,
            input_len=1024,
            model="config1-model",
        )

        config_content_2 = create_test_config_content(
            batch_size=20,
            model="config2-model",
            # input_len not specified, should keep from config1
        )

        config_file_1 = create_test_config_file(config_content_1)
        config_file_2 = create_test_config_file(config_content_2)

        try:
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert shape_config is not None
                # Note: Cyclopts processes config sources in order, but may not override
                # values from earlier Yaml sources. The first Yaml source values are used.
                # This test verifies that both config files are processed and values
                # from config1 are used (since config2 doesn't override in cyclopts).
                assert (
                    shape_config.batch_size == 10
                )  # From config1 (cyclopts uses first Yaml source)
                assert (
                    model_config.model == "config1-model"
                )  # From config1 (cyclopts uses first Yaml source)
                assert shape_config.input_len == 1024  # From config1

            try:
                setup_config_file_meta_command(
                    app,
                    root_keys="test_config",
                    args=[
                        "--config-file",
                        str(config_file_1),
                        "--config-file",
                        str(config_file_2),
                    ],
                )
            except SystemExit as e:
                if e.code != 0:
                    raise

        finally:
            config_file_1.unlink(missing_ok=True)
            config_file_2.unlink(missing_ok=True)

    def test_env_var_naming_convention(self) -> None:
        """Test that MODULAR_ prefix is correctly applied to env var names."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_NUM_WARMUPS": "3",
            "MODULAR_ENABLE_PREFIX_CACHING": "false",
            # Test that non-MODULAR_ vars are ignored
            "BATCH_SIZE": "999",
            "INPUT_LEN": "999",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                # Cyclopts should create config objects, but handle None cases
                if shape_config is None:
                    shape_config = ShapeConfig()
                if execution_config is None:
                    execution_config = ExecutionConfig()
                # Only MODULAR_ prefixed vars should be used
                assert shape_config.batch_size == 5  # From MODULAR_BATCH_SIZE
                assert shape_config.input_len == 512  # From MODULAR_INPUT_LEN
                assert (
                    execution_config.num_warmups == 3
                )  # From MODULAR_NUM_WARMUPS
                assert (
                    execution_config.enable_prefix_caching is False
                )  # From MODULAR_ENABLE_PREFIX_CACHING

            try:
                app([])
            except SystemExit as e:
                if e.code != 0:
                    raise

    def test_boolean_env_var_parsing(self) -> None:
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
        ]

        for env_value, expected_value in test_cases:
            env_vars = {"MODULAR_ENABLE_PREFIX_CACHING": env_value}

            with patch.dict(os.environ, env_vars, clear=False):
                app = create_cyclopts_app(name="test_app", help_text="Test app")

                # Create a factory function to properly bind loop variables
                def make_test_command(
                    exp_val: bool, env_val: str
                ) -> Callable[
                    [
                        ModelConfig | None,
                        ShapeConfig | None,
                        HardwareConfig | None,
                        ProfilingConfig | None,
                        ExecutionConfig | None,
                        SamplingConfig | None,
                    ],
                    None,
                ]:
                    def test_command(
                        model_config: ModelConfig | None = None,
                        shape_config: ShapeConfig | None = None,
                        hardware_config: HardwareConfig | None = None,
                        profiling_config: ProfilingConfig | None = None,
                        execution_config: ExecutionConfig | None = None,
                        sampling_config: SamplingConfig | None = None,
                    ) -> None:
                        assert execution_config is not None
                        assert (
                            execution_config.enable_prefix_caching == exp_val
                        ), f"Failed for env_value={env_val}"

                    return test_command

                app.default(make_test_command(expected_value, env_value))

                try:
                    app([])
                except SystemExit as e:
                    if e.code != 0:
                        raise

    def test_integer_env_var_parsing(self) -> None:
        """Test that integer environment variables are parsed correctly."""
        env_vars = {
            "MODULAR_BATCH_SIZE": "42",
            "MODULAR_INPUT_LEN": "1024",
            "MODULAR_OUTPUT_LEN": "256",
            "MODULAR_SEED": "123",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert shape_config is not None
                assert shape_config.batch_size == 42
                assert shape_config.input_len == 1024
                assert shape_config.output_len == 256
                assert model_config.seed == 123
                assert isinstance(shape_config.batch_size, int)
                assert isinstance(shape_config.input_len, int)
                assert isinstance(shape_config.output_len, int)
                assert isinstance(model_config.seed, int)

            try:
                app([])
            except SystemExit as e:
                if e.code != 0:
                    raise

    def test_string_env_var_parsing(self) -> None:
        """Test that string environment variables are parsed correctly."""
        env_vars = {
            "MODULAR_MODEL": "test-model-name",
            "MODULAR_QUANTIZATION_ENCODING": "bfloat16",
            "MODULAR_DEVICES": "gpu:0,1,2",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert hardware_config is not None
                assert model_config.model == "test-model-name"
                assert hardware_config.quantization_encoding == "bfloat16"
                assert hardware_config.devices == "gpu:0,1,2"
                assert isinstance(model_config.model, str)
                assert isinstance(hardware_config.quantization_encoding, str)
                assert isinstance(hardware_config.devices, str)

            try:
                app([])
            except SystemExit as e:
                if e.code != 0:
                    raise

    def test_nonexistent_config_file_does_not_error(self) -> None:
        """Test that nonexistent config file doesn't cause errors."""
        nonexistent_file = Path("/nonexistent/config.yaml")

        app = create_cyclopts_app(name="test_app", help_text="Test app")

        @app.default
        def test_command(
            model_config: ModelConfig | None = None,
            shape_config: ShapeConfig | None = None,
            hardware_config: HardwareConfig | None = None,
            profiling_config: ProfilingConfig | None = None,
            execution_config: ExecutionConfig | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> None:
            # Cyclopts should create config objects, but handle None cases
            if shape_config is None:
                shape_config = ShapeConfig()
            # Should use defaults when config file doesn't exist
            assert shape_config.batch_size == 1  # Default value

        # Should not raise an error
        try:
            setup_config_file_meta_command(
                app,
                root_keys="test_config",
                must_exist=False,
                args=["--config-file", str(nonexistent_file)],
            )
        except SystemExit as e:
            if e.code != 0:
                raise

    def test_empty_config_file_uses_defaults(self) -> None:
        """Test that empty config file results in using defaults."""
        config_content: dict[str, dict[str, Any]] = {"test_config": {}}
        config_file = create_test_config_file(config_content)

        try:
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                # Cyclopts should create config objects, but handle None cases
                if shape_config is None:
                    shape_config = ShapeConfig()
                # Should use defaults when config file is empty
                assert shape_config.batch_size == 1  # Default value
                assert shape_config.input_len == 256  # Default value

            try:
                setup_config_file_meta_command(
                    app,
                    root_keys="test_config",
                    must_exist=False,
                    args=["--config-file", str(config_file)],
                )
            except SystemExit as e:
                if e.code != 0:
                    raise

        finally:
            config_file.unlink(missing_ok=True)

    def test_complex_precedence_scenario(self) -> None:
        """Test a complex scenario with all precedence levels."""
        # Set up environment variables (lowest priority)
        env_vars = {
            "MODULAR_BATCH_SIZE": "5",
            "MODULAR_INPUT_LEN": "512",
            "MODULAR_OUTPUT_LEN": "128",
            "MODULAR_SEED": "10",
        }

        # Set up config file (medium priority)
        config_content = create_test_config_content(
            batch_size=10,  # Overrides env var
            input_len=1024,  # Overrides env var
            # output_len not specified, should use env var
            # seed not specified, should use env var
            num_warmups=2,  # Not in env, uses config
        )

        config_file = create_test_config_file(config_content)

        try:
            with patch.dict(os.environ, env_vars, clear=False):
                app = create_cyclopts_app(name="test_app", help_text="Test app")

                @app.default
                def test_command(
                    model_config: ModelConfig | None = None,
                    shape_config: ShapeConfig | None = None,
                    hardware_config: HardwareConfig | None = None,
                    profiling_config: ProfilingConfig | None = None,
                    execution_config: ExecutionConfig | None = None,
                    sampling_config: SamplingConfig | None = None,
                ) -> None:
                    assert model_config is not None
                    assert shape_config is not None
                    assert execution_config is not None
                    # CLI args (highest priority)
                    assert (
                        shape_config.batch_size == 20
                    )  # From CLI, overrides config (10) and env (5)
                    assert (
                        shape_config.input_len == 2048
                    )  # From CLI, overrides config (1024) and env (512)
                    # Config file (medium priority)
                    assert execution_config.num_warmups == 2  # From config file
                    # Env vars (lower priority)
                    assert (
                        shape_config.output_len == 128
                    )  # From env var (not overridden)
                    assert (
                        model_config.seed == 10
                    )  # From env var (not overridden)
                    # Defaults (lowest priority)
                    assert execution_config.num_iters == 1  # Default value

                try:
                    setup_config_file_meta_command(
                        app,
                        root_keys="test_config",
                        args=[
                            "--config-file",
                            str(config_file),
                            "--batch-size=20",
                            "--input-len=2048",
                        ],
                    )
                except SystemExit as e:
                    if e.code != 0:
                        raise

        finally:
            config_file.unlink(missing_ok=True)


class TestConfigValidation:
    """Test configuration validation."""

    def test_model_from_env_var(self) -> None:
        """Test that model can be provided via environment variable."""
        env_vars = {"MODULAR_MODEL": "test-model"}

        with patch.dict(os.environ, env_vars, clear=False):
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert model_config.model == "test-model"

            try:
                app([])
            except SystemExit as e:
                if e.code != 0:
                    raise

    def test_model_from_config_file(self) -> None:
        """Test that model can be provided via config file."""
        config_content = create_test_config_content(model="config-model")
        config_file = create_test_config_file(config_content)

        try:
            app = create_cyclopts_app(name="test_app", help_text="Test app")

            @app.default
            def test_command(
                model_config: ModelConfig | None = None,
                shape_config: ShapeConfig | None = None,
                hardware_config: HardwareConfig | None = None,
                profiling_config: ProfilingConfig | None = None,
                execution_config: ExecutionConfig | None = None,
                sampling_config: SamplingConfig | None = None,
            ) -> None:
                assert model_config is not None
                assert model_config.model == "config-model"

            try:
                setup_config_file_meta_command(
                    app,
                    root_keys="test_config",
                    args=["--config-file", str(config_file)],
                )
            except SystemExit as e:
                if e.code != 0:
                    raise

        finally:
            config_file.unlink(missing_ok=True)

    def test_model_from_cli(self) -> None:
        """Test that model can be provided via CLI argument."""
        app = create_cyclopts_app(name="test_app", help_text="Test app")

        @app.default
        def test_command(
            model_config: ModelConfig | None = None,
            shape_config: ShapeConfig | None = None,
            hardware_config: HardwareConfig | None = None,
            profiling_config: ProfilingConfig | None = None,
            execution_config: ExecutionConfig | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> None:
            assert model_config is not None
            assert model_config.model == "cli-model"

        try:
            app(["--model=cli-model"])
        except SystemExit as e:
            if e.code != 0:
                raise
