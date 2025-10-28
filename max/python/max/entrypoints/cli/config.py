# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


"""Utilities for working with Config objects in Click."""

from __future__ import annotations

import functools
import inspect
import json
import pathlib
from collections.abc import Callable
from dataclasses import MISSING, Field, fields
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import click
from max.driver import DeviceSpec
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    MAXConfig,
    MAXModelConfig,
    PipelineConfig,
    ProfilingConfig,
    SamplingConfig,
)
from typing_extensions import ParamSpec

from .device_options import DevicesOptionType

VALID_CONFIG_TYPES = [str, bool, Enum, Path, DeviceSpec, int, float, dict]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class JSONType(click.ParamType):
    """Click parameter type for JSON input."""

    name = "json"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> Any:
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}", param, ctx)


def get_interior_type(type_hint: type | str | Any) -> type[Any]:
    interior_args = set(get_args(type_hint)) - set([type(None)])
    if len(interior_args) > 1:
        raise ValueError(
            "Parsing does not currently supported Union type, with more than"
            f" one non-None type: {type_hint}"
        )

    return get_args(type_hint)[0]


def is_optional(type_hint: type | str | Any) -> bool:
    return get_origin(type_hint) in (Union, UnionType) and type(
        None
    ) in get_args(type_hint)


def is_flag(field_type: Any) -> bool:
    return field_type is bool


def validate_field_type(field_type: Any) -> bool:
    if is_optional(field_type):
        test_type = get_args(field_type)[0]
    elif get_origin(field_type) is list:
        test_type = get_interior_type(field_type)
    else:
        test_type = field_type

    if get_origin(test_type) is dict:
        return True

    for valid_type in VALID_CONFIG_TYPES:
        if valid_type == test_type:
            return True

        if get_origin(valid_type) is None and inspect.isclass(test_type):
            if issubclass(test_type, valid_type):
                return True
    raise ValueError(f"type '{test_type}' not supported in config.")


def get_field_type(field_type: Any):  # noqa: ANN201
    validate_field_type(field_type)

    # Get underlying core field type, is Optional or list.
    if is_optional(field_type):
        field_type = get_interior_type(field_type)
    elif get_origin(field_type) is list:
        field_type = get_interior_type(field_type)

    # Update the field_type to be format specific.
    if field_type == Path:
        field_type = click.Path(path_type=pathlib.Path)
    elif get_origin(field_type) is dict or field_type is dict:
        field_type = JSONType()
    elif inspect.isclass(field_type):
        if issubclass(field_type, Enum):
            field_type = click.Choice(list(field_type), case_sensitive=False)

    return field_type


def get_default(dataclass_field: Field[Any]) -> Any:
    if dataclass_field.default_factory != MISSING:
        default = dataclass_field.default_factory()
    elif dataclass_field.default != MISSING:
        default = dataclass_field.default
    else:
        default = None

    return default


def is_multiple(field_type: Any) -> bool:
    return get_origin(field_type) is list


def get_normalized_flag_name(
    dataclass_field: Field[Any], field_type: Any
) -> str:
    normalized_name = dataclass_field.name.lower().replace("_", "-")

    if is_flag(field_type):
        return f"--{normalized_name}/--no-{normalized_name}"
    else:
        return f"--{normalized_name}"


def create_click_option(
    help_for_fields: dict[str, str],
    dataclass_field: Field[Any],
    field_type: Any,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # Get Help text.
    help_text = help_for_fields.get(dataclass_field.name, None)

    # Get help field.
    return click.option(
        get_normalized_flag_name(dataclass_field, field_type),
        show_default=True,
        help=help_text,
        is_flag=is_flag(field_type),
        default=get_default(dataclass_field),
        multiple=is_multiple(field_type),
        type=get_field_type(field_type),
    )


def config_to_flag(
    cls: type[MAXConfig], prefix: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    options = []
    if hasattr(cls, "help"):
        help_text = cls.help()
    else:
        help_text = {}
    field_types = get_type_hints(cls)
    for _field in fields(cls):
        # Skip private config fields.
        if _field.name.startswith("_") or _field.name in (
            "device_specs",
            "in_dtype",
            "out_dtype",
            "pdl_level",
        ):
            continue

        if prefix:
            field_type = field_types[_field.name]
            new_name = f"{prefix}_{_field.name}"

            if _field.name in help_text:
                help_text[new_name] = help_text[_field.name]

            _field.name = new_name
            new_option = create_click_option(help_text, _field, field_type)
        else:
            new_option = create_click_option(
                help_text, _field, field_types[_field.name]
            )
        options.append(new_option)

    def apply_flags(func: Callable[_P, _R]) -> Callable[_P, _R]:
        for option in reversed(options):
            func = option(func)
        return func

    return apply_flags


def pipeline_config_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # The order of these decorators must be preserved - ie. PipelineConfig
    # must be applied only after KVCacheConfig, ProfilingConfig etc.
    @config_to_flag(PipelineConfig)
    @config_to_flag(MAXModelConfig)
    @config_to_flag(MAXModelConfig, prefix="draft")
    @config_to_flag(KVCacheConfig)
    @config_to_flag(LoRAConfig)
    @config_to_flag(ProfilingConfig)
    @config_to_flag(SamplingConfig)
    @click.option(
        "--devices",
        is_flag=False,
        type=DevicesOptionType(),
        show_default=False,
        default="default",
        help=(
            "Whether to run the model on CPU (--devices=cpu), GPU (--devices=gpu)"
            " or a list of GPUs (--devices=gpu:0,1) etc. An ID value can be"
            " provided optionally to indicate the device ID to target. If not"
            " provided, the model will run on the first available GPU (--devices=gpu),"
            " or CPU if no GPUs are available (--devices=cpu)."
        ),
    )
    @click.option(
        "--draft-devices",
        is_flag=False,
        type=DevicesOptionType(),
        show_default=False,
        default="default",
        help=(
            "Whether to run the model on CPU (--devices=cpu), GPU (--devices=gpu)"
            " or a list of GPUs (--devices=gpu:0,1) etc. An ID value can be"
            " provided optionally to indicate the device ID to target. If not"
            " provided, the model will run on the first available GPU (--devices=gpu),"
            " or CPU if no GPUs are available (--devices=cpu)."
        ),
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        def is_str_or_list_of_int(value: Any) -> TypeGuard[str | list[int]]:
            return isinstance(value, str) or (
                isinstance(value, list)
                and all(isinstance(x, int) for x in value)
            )

        # Remove the options from kwargs and replace with unified device_specs.
        devices = kwargs.pop("devices")
        draft_devices = kwargs.pop("draft_devices")
        assert is_str_or_list_of_int(devices)
        assert is_str_or_list_of_int(draft_devices)

        kwargs["device_specs"] = DevicesOptionType.device_specs(devices)
        kwargs["draft_device_specs"] = DevicesOptionType.device_specs(
            draft_devices
        )

        return func(*args, **kwargs)

    return wrapper


def sampling_params_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @click.option(
        "--top-k",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Limits the sampling to the K most probable tokens. This defaults to 255. For greedy sampling, set to 1.",
    )
    @click.option(
        "--top-p",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="Only use the tokens whose cumulative probability is within the top_p threshold. This applies to the top_k tokens.",
    )
    @click.option(
        "--min-p",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this.",
    )
    @click.option(
        "--temperature",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="Controls the randomness of the model's output; higher values produce more diverse responses.",
    )
    @click.option(
        "--frequency-penalty",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="The frequency penalty to apply to the model's output. A positive value will penalize new tokens based on their frequency in the generated text.",
    )
    @click.option(
        "--presence-penalty",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="The presence penalty to apply to the model's output. A positive value will penalize new tokens that have already appeared in the generated text at least once.",
    )
    @click.option(
        "--repetition-penalty",
        is_flag=False,
        type=float,
        show_default=False,
        default=None,
        help="The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens that have already appeared in the generated text at least once.",
    )
    @click.option(
        "--max-new-tokens",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Maximum number of new tokens to generate during a single inference pass of the model.",
    )
    @click.option(
        "--min-new-tokens",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Minimum number of tokens to generate in the response.",
    )
    @click.option(
        "--ignore-eos",
        is_flag=True,
        show_default=False,
        default=None,
        help="If True, the response will ignore the EOS token, and continue to generate until the max tokens or a stop string is hit.",
    )
    @click.option(
        "--stop",
        is_flag=False,
        type=str,
        show_default=False,
        default=None,
        multiple=True,
        help="A list of detokenized sequences that can be used as stop criteria when generating a new sequence. Can be specified multiple times.",
    )
    @click.option(
        "--stop-token-ids",
        is_flag=False,
        type=str,
        show_default=False,
        default=None,
        help="A list of token ids that are used as stopping criteria when generating a new sequence. Comma-separated integers.",
    )
    @click.option(
        "--detokenize/--no-detokenize",
        is_flag=True,
        show_default=False,
        default=None,
        help="Whether to detokenize the output tokens into text.",
    )
    @click.option(
        "--seed",
        is_flag=False,
        type=int,
        show_default=False,
        default=None,
        help="Seed for the random number generator.",
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return func(*args, **kwargs)

    return wrapper


def parse_task_flags(task_flags: tuple[str, ...]) -> dict[str, str]:
    """Parse task flags into a dictionary.

    The flags must be in the format `flag_name=flag_value`.

    This requires that the task flags are:
    1. Passed and interpreted as strings, including their values.
    2. Be passed as a list of strings via explicit --task-arg flags. For example:
        --task-arg=flag1=value1 --task-arg=flag2=value2

    Args:
        task_flags: A tuple of task flags.

    Returns:
        A dictionary of parsed flag values.
    """
    flags = {}
    for flag in task_flags:
        if "=" not in flag or flag.startswith("--"):
            raise ValueError(
                f"Flag must be in format 'flag_name=flag_value', got: {flag}"
            )

        flag_name, flag_value = flag.split("=", 1)
        flags[flag_name.replace("-", "_")] = flag_value
    return flags
