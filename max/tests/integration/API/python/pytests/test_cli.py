# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import click
from click.testing import CliRunner
import enum
import pytest
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional
from cli.config import (
    get_field_type,
    validate_field,
    get_default,
    is_flag,
    is_multiple,
    pipeline_config_options,
    is_optional,
)
from max.driver import DeviceSpec
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
    device_spec_field: DeviceSpec = DeviceSpec.cpu()
    optional_str_field: Optional[str] = None


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
    "device_spec_field": Output(
        default=DeviceSpec.cpu(),
        field_type=DeviceSpec,
        flag=False,
        multiple=False,
        optional=False,
    ),
    "optional_str_field": Output(
        default=None,
        field_type=str,
        flag=False,
        multiple=False,
        optional=True,
    ),
}


def test_pipeline_config_cli_parsing():
    for config_field in fields(PipelineConfig):
        if not config_field.name.startswith("_"):
            validate_field(config_field)


def test_cli__get_default():
    for config_field in fields(TestConfig):
        assert (
            get_default(config_field)
            == VALID_RESULTS[config_field.name].default
        )


def test_cli__get_field_type():
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            field_type = get_field_type(config_field)

            # If optional, ensure the underlying value has the correct type
            expected_type = VALID_RESULTS[config_field.name].field_type
            if not isinstance(expected_type, type):
                assert type(expected_type) is type(field_type)
            else:
                assert expected_type == field_type


def test_cli__option_is_flag():
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            flag = is_flag(config_field)
            assert (
                flag == VALID_RESULTS[config_field.name].flag
            ), f"failed test_is_flag for {config_field.name}"


def test_cli__option_is_multiple():
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            multiple = is_multiple(config_field)
            assert multiple == VALID_RESULTS[config_field.name].multiple


def test_cli__option_is_optional():
    for config_field in fields(TestConfig):
        if not config_field.name.startswith("_"):
            optional = is_optional(config_field.type)
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
        args=["--max-length", "10"], expected={"max_length": 10}, valid=False
    ),
    TestCommand(args=[], expected={}, valid=False),
    TestCommand(
        args=[
            "--huggingface-repo-id",
            "modularai/test_modelid",
            "--trust-remote-code",
        ],
        expected={
            "huggingface_repo_id": "modularai/test_modelid",
            "trust_remote_code": True,
        },
        valid=False,
    ),
    TestCommand(
        args=[
            "--trust-remote-code",
        ],
        expected={
            "trust_remote_code": True,
        },
        valid=False,
    ),
    TestCommand(
        args=[
            "--architecture",
            "LlamaForCausalLM",
            "--max-length",
            "10",
        ],
        expected={
            "trust_remote_code": False,
            "max_length": 10,
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--architecture",
            "LlamaForCausalLM",
            "--cache-strategy",
            "naive",
        ],
        expected={
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.NAIVE,
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--architecture",
            "LlamaForCausalLM",
            "--max-length",
            "10",
        ],
        expected={
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.CONTINUOUS,
            "max_length": 10,
        },
        valid=True,
    ),
    TestCommand(
        args=[
            "--architecture",
            "LlamaForCausalLM",
            "--max-length",
            "10",
        ],
        expected={
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.CONTINUOUS,
            "max_length": 10,
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
            "cache_strategy": KVCacheStrategy.CONTINUOUS,
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
            "--use-gpu",
        ],
        expected={
            "huggingface_repo_id": "modularai/llama-3.1",
            "trust_remote_code": False,
            "cache_strategy": KVCacheStrategy.CONTINUOUS,
            "weight_path": [
                Path("model1.safetensors"),
                Path("model2.safetensors"),
                Path("model3.safetensors"),
            ],
            "device_spec": DeviceSpec.cuda(id=0),
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
