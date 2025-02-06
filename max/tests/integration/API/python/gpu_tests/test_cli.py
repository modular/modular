# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import enum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, get_type_hints

import click
import pytest
from click.testing import CliRunner
from max.driver import DeviceSpec
from max.entrypoints.cli import (
    get_default,
    get_field_type,
    is_flag,
    is_multiple,
    is_optional,
    pipeline_config_options,
    validate_field_type,
)
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import KVCacheStrategy


class TestEnum(str, enum.Enum):
    DEFAULT = "default"
    ALT = "alt"


@dataclass
class TestConfig:
    bool_field: bool = False
    enum_field: TestEnum = TestEnum.DEFAULT
    path_sequence_field: list[Path] = field(default_factory=list)
    device_specs_field: list[DeviceSpec] = field(
        default_factory=lambda: [DeviceSpec.cpu()]
    )
    optional_str_field: Optional[str] = None
    optional_enum_field: Optional[TestEnum] = None


@dataclass
class Output:
    default: Any
    field_type: Any
    flag: bool
    multiple: bool
    optional: bool


VALID_RESULTS = {
    "bool_field": Output(
        default=False,
        field_type=bool,
        flag=True,
        multiple=False,
        optional=False,
    ),
    "enum_field": Output(
        default=TestEnum.DEFAULT,
        field_type=click.Choice([e for e in TestEnum]),
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
        field_type=click.Choice([e for e in TestEnum]),
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
    for config_field in fields(TestConfig):
        assert (
            get_default(config_field)
            == VALID_RESULTS[config_field.name].default
        )


def test_cli__get_field_type():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            field_type = get_field_type(field_types[config_field.name])

            # If optional, ensure the underlying value has the correct type
            expected_type = VALID_RESULTS[config_field.name].field_type
            if not isinstance(expected_type, type):
                assert type(expected_type) is type(field_type)
            else:
                assert expected_type == field_type


def test_cli__option_is_flag():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            flag = is_flag(field_types[config_field.name])
            assert flag == VALID_RESULTS[config_field.name].flag, (
                f"failed test_is_flag for {config_field.name}"
            )


def test_cli__option_is_multiple():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            multiple = is_multiple(field_types[config_field.name])
            assert multiple == VALID_RESULTS[config_field.name].multiple


def test_cli__option_is_optional():
    field_types = get_type_hints(TestConfig)
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            optional = is_optional(field_types[config_field.name])
            assert optional == VALID_RESULTS[config_field.name].optional


@click.group()
def cli():
    pass


@dataclass
class TestCommand:
    args: list[str]
    expected: dict[str, Any]
    valid: bool


VALID_COMMANDS = [
    TestCommand(
        args=[
            "--huggingface-repo-id",
            "modularai/replit-code-1.5",
            "--cache-strategy",
            "naive",
            "--devices",
            "gpu",
        ],
        expected={
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.NAIVE,
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--huggingface-repo-id",
            "modularai/llama-3.1",
            "--weight-path",
            "model1.safetensors",
            "--weight-path",
            "model2.safetensors",
            "--weight-path",
            "model3.safetensors",
        ],
        expected={
            "huggingface_repo_id": "modularai/llama-3.1",
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.MODEL_DEFAULT,
            "weight_path": [
                Path("model1.safetensors"),
                Path("model2.safetensors"),
                Path("model3.safetensors"),
            ],
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--huggingface-repo-id",
            "modularai/llama-3.1",
            "--weight-path",
            "model1.safetensors",
            "--weight-path",
            "model2.safetensors",
            "--weight-path",
            "model3.safetensors",
            "--devices",
            "gpu-0",
        ],
        expected={
            "huggingface_repo_id": "modularai/llama-3.1",
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.MODEL_DEFAULT,
            "weight_path": [
                Path("model1.safetensors"),
                Path("model2.safetensors"),
                Path("model3.safetensors"),
            ],
            "device_specs": [DeviceSpec.accelerator(id=0)],
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
    test_command = VALID_COMMANDS[idx]

    if test_command.valid:
        # Initialize Pipeline Config.
        pipeline_config = PipelineConfig(**config_kwargs)

        for attr_name, expected_value in test_command.expected.items():
            assert hasattr(pipeline_config, attr_name)
            test_value = getattr(pipeline_config, attr_name)
            assert test_value == expected_value

    else:
        with pytest.raises(Exception):
            # Initialize Pipeline Config.
            pipeline_config = PipelineConfig(**config_kwargs)

            for attr_name, expected_value in test_command.expected.items():
                assert hasattr(pipeline_config, attr_name)
                test_value = getattr(pipeline_config, attr_name)
                assert test_value == expected_value


def test_cli__terminal_commands():
    runner = CliRunner()
    for idx, command in enumerate(VALID_COMMANDS):
        command.args.extend(["--idx", str(idx)])
        print(f"full_args: {command.args}")
        result = runner.invoke(testing, command.args)

        assert result.exit_code == 0
