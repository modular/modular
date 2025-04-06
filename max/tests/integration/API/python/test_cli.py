# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from max.entrypoints.cli import pipeline_config_options
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import KVCacheStrategy
from test_common.pipeline_cli_utils import CLITestCommand
from test_common.pipeline_model import mock_estimate_memory_footprint


@click.group()
def cli():
    pass


TEST_COMMANDS = [
    CLITestCommand(
        args=["--max-length", "10", "--devices", "cpu"],
        expected={"max_length": 10},
        valid=False,
    ),
    CLITestCommand(
        args=["--devices", "cpu"],
        expected={},
        valid=False,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/test_modelid",
            "--trust-remote-code",
            "--devices",
            "cpu",
        ],
        expected={
            "model_config": {
                "model_path": "modularai/test_modelid",
                "trust_remote_code": True,
            },
        },
        valid=False,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--trust-remote-code",
            "--devices",
            "cpu",
        ],
        expected={
            "model_config": {
                "trust_remote_code": True,
            },
        },
        valid=True,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--max-length",
            "10",
            "--devices",
            "cpu",
        ],
        expected={
            "model_config": {
                "trust_remote_code": False,
            },
            "max_length": 10,
        },
        valid=True,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--weight-path",
            "model1.safetensors",
            "--weight-path",
            "model2.safetensors",
            "--weight-path",
            "model3.safetensors",
            "--devices",
            "cpu",
        ],
        expected={
            "model_config": {
                "model_path": "modularai/llama-3.1",
                "trust_remote_code": False,
                "weight_path": [
                    Path("model1.safetensors"),
                    Path("model2.safetensors"),
                    Path("model3.safetensors"),
                ],
                "kv_cache_config": {
                    "cache_strategy": KVCacheStrategy.MODEL_DEFAULT,
                },
            },
        },
        valid=True,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--cache-strategy",
            "naive",
            "--devices",
            "cpu",
        ],
        expected={
            "model_config": {
                "model_path": "modularai/llama-3.1",
                "trust_remote_code": False,
                "kv_cache_config": {
                    "cache_strategy": KVCacheStrategy.NAIVE,
                },
            },
        },
        valid=True,
    ),
]


@cli.command(name="testing")
@pipeline_config_options
@click.option(
    "--idx",
    type=int,
)
def testing(
    idx,
    **config_kwargs,
):
    # Retrieve test command.
    test_command = TEST_COMMANDS[idx]

    if test_command.valid:
        # Initialize Pipeline Config.
        pipeline_config = PipelineConfig(**config_kwargs)

        for attr_name, expected_value in test_command.expected.items():
            assert hasattr(pipeline_config, attr_name)
            test_value = getattr(pipeline_config, attr_name)

            if isinstance(expected_value, dict):
                # Recursively check nested dictionaries
                for key, value in expected_value.items():
                    assert hasattr(test_value, key)
                    nested_test_value = getattr(test_value, key)

                    if key == "kv_cache_config":
                        # Handle nested kv_cache_config
                        for kv_key, kv_value in value.items():
                            assert hasattr(nested_test_value, kv_key)
                            kv_test_value = getattr(nested_test_value, kv_key)
                            assert kv_test_value == kv_value
                    else:
                        assert nested_test_value == value
            else:
                assert test_value == expected_value

    else:
        with pytest.raises(Exception):
            # Initialize Pipeline Config.
            pipeline_config = PipelineConfig(**config_kwargs)

            for attr_name, expected_value in test_command.expected.items():
                assert hasattr(pipeline_config, attr_name)
                test_value = getattr(pipeline_config, attr_name)

                if isinstance(expected_value, dict):
                    # Recursively check nested dictionaries
                    for key, value in expected_value.items():
                        assert hasattr(test_value, key)
                        nested_test_value = getattr(test_value, key)

                        if key == "kv_cache_config":
                            # Handle nested kv_cache_config
                            for kv_key, kv_value in value.items():
                                assert hasattr(nested_test_value, kv_key)
                                kv_test_value = getattr(
                                    nested_test_value, kv_key
                                )
                                assert kv_test_value == kv_value
                        else:
                            assert nested_test_value == value
                else:
                    assert test_value == expected_value


@pytest.mark.parametrize(
    "command, idx",
    [(cmd, i) for i, cmd in enumerate(TEST_COMMANDS)],
    ids=["TEST_COMMANDS[" + str(i) + "]" for i in range(len(TEST_COMMANDS))],
)
@mock_estimate_memory_footprint
def test_cli__terminal_commands(command, idx):
    """
    Test individual terminal CLI commands

    Args:
        command: Command object containing args and valid flag
        idx: Index of the command in TEST_COMMANDS list
    """
    runner = CliRunner()

    # Add the index to the command args
    command_args = command.args + ["--idx", str(idx)]
    print(f"full_args: {command_args}")

    result = runner.invoke(testing, command_args)
    assert result.exit_code == 0, f"Command failed: {command_args}"
