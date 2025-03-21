# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import click
import pytest
from click.testing import CliRunner
from max.driver import DeviceSpec
from max.engine import GPUProfilingMode
from max.entrypoints.cli import (
    get_default,
    get_field_type,
    is_flag,
    is_multiple,
    is_optional,
    pipeline_config_options,
    validate_field_type,
)
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
)
from test_common.pipeline_cli_utils import (
    CLITestCommand,
    CLITestConfig,
    CLITestEnum,
    Output,
)
from test_common.pipeline_model import DUMMY_ARCH

PIPELINE_REGISTRY.reset()
PIPELINE_REGISTRY.register(DUMMY_ARCH)

VALID_RESULTS = {
    "bool_field": Output(
        default=False,
        field_type=bool,
        flag=True,
        multiple=False,
        optional=False,
    ),
    "enum_field": Output(
        default=CLITestEnum.DEFAULT,
        field_type=click.Choice([e for e in CLITestEnum]),
        flag=False,
        multiple=False,
        optional=False,
    ),
    "path_sequence_field": Output(
        default=[],
        field_type=click.Path(path_type=Path),
        flag=False,
        multiple=True,
        optional=False,
    ),
    "device_specs_field": Output(
        default=[DeviceSpec.cpu()],
        field_type=DeviceSpec,
        flag=False,
        multiple=True,
        optional=False,
    ),
    "optional_str_field": Output(
        default=None,
        field_type=str,
        flag=False,
        multiple=False,
        optional=True,
    ),
    "optional_enum_field": Output(
        default=None,
        field_type=click.Choice([e for e in CLITestEnum]),
        flag=False,
        multiple=False,
        optional=True,
    ),
}


def test_pipeline_config_cli_parsing():
    field_types = get_type_hints(PipelineConfig)
    for config_field in fields(PipelineConfig):
        if not config_field.name.startswith("_"):
            validate_field_type(field_types[config_field.name])


def test_cli__get_default():
    for config_field in fields(CLITestConfig):
        assert (
            get_default(config_field)
            == VALID_RESULTS[config_field.name].default
        )


def test_cli__get_field_type():
    field_types = get_type_hints(CLITestConfig)
    for config_field in fields(CLITestConfig):
        if not config_field.name.startswith("_"):
            field_type = get_field_type(field_types[config_field.name])

            # If optional, ensure the underlying value has the correct type
            expected_type = VALID_RESULTS[config_field.name].field_type
            if not isinstance(expected_type, type):
                assert type(expected_type) is type(field_type)
            else:
                assert expected_type == field_type


def test_cli__option_is_flag():
    field_types = get_type_hints(CLITestConfig)
    for config_field in fields(CLITestConfig):
        if not config_field.name.startswith("_"):
            flag = is_flag(field_types[config_field.name])
            assert flag == VALID_RESULTS[config_field.name].flag, (
                f"failed test_is_flag for {config_field.name}"
            )


def test_cli__option_is_multiple():
    field_types = get_type_hints(CLITestConfig)
    for config_field in fields(CLITestConfig):
        if not config_field.name.startswith("_"):
            multiple = is_multiple(field_types[config_field.name])
            assert multiple == VALID_RESULTS[config_field.name].multiple


def test_cli__option_is_optional():
    field_types = get_type_hints(CLITestConfig)
    for config_field in fields(CLITestConfig):
        if not config_field.name.startswith("_"):
            optional = is_optional(field_types[config_field.name])
            assert optional == VALID_RESULTS[config_field.name].optional


@click.group()
def cli():
    pass


# TODO(AITLIB-277): Add back test coverage for kv_cache_config. These had to be removed
# as the kv cache strategy changes within config right after the config is
# initialized, which may not be what is provided in the command line.
TEST_COMMANDS = [
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--cache-strategy",
            "naive",
            "--devices",
            "gpu",
        ],
        expected={
            "model_config": {
                "trust_remote_code": False,
            },
        },
        valid=True,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--devices",
            "gpu",
            "--gpu-profiling",
            "on",
        ],
        expected={
            "profiling_config": {
                "gpu_profiling": GPUProfilingMode.ON,
            },
        },
        valid=True,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--devices",
            "gpu",
            "--gpu-profiling",
            "detailed",
        ],
        expected={
            "profiling_config": {
                "gpu_profiling": GPUProfilingMode.DETAILED,
            },
        },
        valid=True,
    ),
    CLITestCommand(
        args=[
            "--model-path",
            "modularai/llama-3.1",
            "--devices",
            "gpu",
            "--gpu-profiling",
            "invalid",
        ],
        expected={
            "profiling_config": {
                "gpu_profiling": GPUProfilingMode.DETAILED,
            },
        },
        valid=False,
    ),
    # TODO(AITLIB-278): Add back test coverage for weight_path.
    # CLITestCommand(
    #     args=[
    #         "--model-path",
    #         "modularai/llama-3.1",
    #         "--weight-path",
    #         "model1.safetensors",
    #         "--weight-path",
    #         "model2.safetensors",
    #         "--weight-path",
    #         "model3.safetensors",
    #     ],
    #     expected={
    #         "model_config": {
    #             "model_path": "modularai/llama-3.1",
    #             "trust_remote_code": False,
    #             "weight_path": [
    #                 Path("model1.safetensors"),
    #                 Path("model2.safetensors"),
    #                 Path("model3.safetensors"),
    #             ],
    #         },
    #     },
    #     valid=True,
    # ),
    # CLITestCommand(
    #     args=[
    #         "--model-path",
    #         "modularai/llama-3.1",
    #         "--weight-path",
    #         "model1.safetensors",
    #         "--weight-path",
    #         "model2.safetensors",
    #         "--weight-path",
    #         "model3.safetensors",
    #         "--devices",
    #         "gpu:0",
    #     ],
    #     expected={
    #         "model_config": {
    #             "model_path": "modularai/llama-3.1",
    #             "trust_remote_code": False,
    #             "weight_path": [
    #                 Path("model1.safetensors"),
    #                 Path("model2.safetensors"),
    #                 Path("model3.safetensors"),
    #             ],
    #             "device_specs": [DeviceSpec.accelerator(id=0)],
    #         },
    #     },
    #     valid=True,
    # ),
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


def test_cli__terminal_commands():
    runner = CliRunner()
    for idx, command in enumerate(TEST_COMMANDS):
        command.args.extend(["--idx", str(idx)])
        print(f"full_args: {command.args}")
        result = runner.invoke(testing, command.args)

        if command.valid:
            assert result.exit_code == 0
        else:
            assert result.exit_code != 0
