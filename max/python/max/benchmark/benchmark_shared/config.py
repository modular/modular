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

"""Benchmark configuration classes with inheritance structure for MAX benchmarks."""

import argparse
import logging
import tempfile
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml
from pydantic import Field

from .datasets import DatasetMode, DistributionParameter

logger = logging.getLogger(__name__)

from max.config import ConfigFileModel, deep_merge_max_configs

BaseBackend = Literal[
    "modular",
    "sglang",
    "trtllm",
    "vllm",
]

Backend = Literal[
    "modular",
    "modular-chat",
    "sglang",
    "sglang-chat",
    "trtllm",
    "trtllm-chat",
    "vllm",
    "vllm-chat",
]

Endpoint = Literal[
    "/v1/completions",
    "/v1/chat/completions",
    "/v2/models/ensemble/generate_stream",
    "/v1/responses",
]

CACHE_RESET_ENDPOINT_MAP: Mapping[Backend, str] = {
    "modular": "/reset_prefix_cache",
    "modular-chat": "/reset_prefix_cache",
    "vllm": "/reset_prefix_cache",
    "vllm-chat": "/reset_prefix_cache",
    "sglang": "/flush_cache",
    "sglang-chat": "/flush_cache",
}

BenchmarkTask = Literal[
    "text-generation",
    "text-to-image",
    "image-to-image",
]

PIXEL_GENERATION_TASKS: tuple[BenchmarkTask, ...] = (
    "text-to-image",
    "image-to-image",
)


def _add_config_file_arg_to_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the --config-file argument to a parser.

    Args:
        parser: The parser to add the argument to.
    """
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to configuration file. If provided, this config will inherit from the default config and override its values.",
    )
    return parser


def _resolve_user_provided_config_file_cli_arg(
    args: Sequence[str] | None = None,
) -> tuple[Path | None, list[str]]:
    """Resolve the user-provided --config-file argument from command line arguments.

    This utility function extracts the config file path from command line arguments
    before the main argument parsing, allowing the config file to be loaded and used
    as defaults for the main parser.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.

    Returns:
        Tuple of (config_file_path, remaining_args) where:
        - config_file_path: Path to the config file if provided, None otherwise
        - remaining_args: List of remaining arguments after removing --config-file
    """
    # Create a preliminary parser to get the config file path
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser = _add_config_file_arg_to_parser(preliminary_parser)

    # Parse preliminary args to get config file path
    preliminary_args, remaining_args = preliminary_parser.parse_known_args(
        args=args
    )
    return preliminary_args.config_file, remaining_args


def _resolve_argparse_type(
    field_type: Any,
) -> tuple[Any, str | type[argparse.Action] | None]:
    """Determine the appropriate argparse type and action for a type annotation.

    Args:
        field_type: The type annotation to analyze.

    Returns:
        Tuple of (type_func, action) for argparse.add_argument().
    """
    origin = get_origin(field_type)
    type_args = get_args(field_type)

    is_union = origin is Union
    if not is_union and origin is not None and hasattr(types, "UnionType"):
        is_union = origin is types.UnionType

    if is_union:
        non_none = [a for a in type_args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_argparse_type(non_none[0])
        return str, None

    if origin is list:
        if type_args and type_args[0] in (int, float, str):
            return type_args[0], None
        return str, None

    if field_type in (int, float, str):
        return field_type, None

    if field_type is bool:
        return None, argparse.BooleanOptionalAction

    return str, None


class HardwareConfig(ConfigFileModel):
    """Configuration class for hardware options."""

    devices: str | None = Field(default=None)
    """Hardware device on which model will be executed. Valid values: 'cpu', 'gpu', 'gpu:0,1,2'."""


class SamplingConfig(ConfigFileModel):
    """Configuration class for sampling options."""

    top_k: int | None = Field(default=None)
    """Limits the sampling to the K most probable tokens. Default: None (no sampling)."""


class BenchmarkCommonConfig(ConfigFileModel):
    tokenizer: str | None = None
    """Name or path of the tokenizer, if not using the default tokenizer."""

    model_max_length: int | None = None
    """Override for tokenizer max length. Needed if server has a lower max length than the tokenizer."""

    trust_remote_code: bool = False
    """Trust remote code from huggingface."""

    # Dataset configuration (common across all benchmark types)
    dataset_name: str | None = None
    """Name of the dataset to benchmark on."""

    dataset_path: str | None = None
    """Path to the dataset."""

    dataset_mode: DatasetMode = "huggingface"
    """Mode for loading the dataset: LOCAL (from local path/env var) or HUGGINGFACE (HuggingFace Hub)."""

    # Basic workload parameters
    num_prompts: int | None = None
    """Number of prompts to process."""

    seed: int = 0
    """Random seed for reproducibility."""

    # Control flags
    disable_tqdm: bool = False
    """Specify to disable tqdm progress bar."""

    print_inputs_and_outputs: bool = False
    """Print all input and outputs to console."""


class BaseBenchmarkConfig(ConfigFileModel):
    """Base configuration class containing parameters common to all benchmark types.

    This class contains the core parameters that are shared across all benchmark types:
    - Model and tokenizer configuration
    - Basic dataset configuration
    - Common workload parameters
    - Basic output control
    - Result saving configuration
    - Common control flags
    """

    _config_file_section_name: ClassVar[str] = "benchmark_config"
    """The section name to use when loading this config from a config file."""

    section_name: str | None = Field(default="benchmark_config", exclude=True)
    """Default section name for benchmark config files.

    Overrides ConfigFileModel's default to automatically extract the
    ``benchmark_config`` section when loading YAML config files."""

    # Model and tokenizer configuration (common to all benchmarks)
    model: str | None = Field(
        default=None,
        description="Name of the model. Required when running benchmark.",
    )

    tokenizer: str | None = Field(
        default=None,
        description="Name or path of the tokenizer, if not using the default tokenizer.",
    )

    model_max_length: int | None = Field(
        default=None,
        description="Override for tokenizer max length. Needed if server has a lower max length than the tokenizer.",
    )

    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code from huggingface.",
    )

    # Dataset configuration (common across all benchmark types)
    dataset_name: str = Field(
        default="sharegpt",
        description="Name of the dataset to benchmark on.",
    )

    dataset_path: str | None = Field(
        default=None,
        description="Path to the dataset.",
    )

    dataset_mode: DatasetMode = Field(
        default="huggingface",
        description="Mode for loading the dataset: LOCAL (from local path/env var) or HUGGINGFACE (HuggingFace Hub).",
    )

    # Basic workload parameters
    num_prompts: int | None = Field(
        default=None,
        description="Number of prompts to process.",
    )

    seed: int = Field(
        default=0,
        description="Random seed for reproducibility.",
    )

    # Control flags
    disable_tqdm: bool = Field(
        default=False,
        description="Specify to disable tqdm progress bar.",
    )

    print_inputs_and_outputs: bool = Field(
        default=False,
        description="Print all input and outputs to console.",
    )

    # TODO: This can be removed once we're on cyclopts.
    @classmethod
    def help(cls) -> dict[str, str]:
        """Build help dictionary from pydantic field descriptions.

        Returns:
            Dictionary of config options and their descriptions.
        """
        return {
            name: field_info.description
            for name, field_info in cls.model_fields.items()
            if field_info.description
        }

    @classmethod
    def get_default_required_fields(cls) -> set[str]:
        """Get required fields for the benchmark config."""
        return {"model"}

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        section_name: str | None = None,
    ) -> "BaseBenchmarkConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.
            section_name: Optional section name override.

        Returns:
            An instance of this config class populated from the file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError(
                "Configuration file must contain a dictionary at the top level"
            )

        section = section_name or cls._config_file_section_name
        if section in config_dict:
            config_data = config_dict[section]
            if not isinstance(config_data, dict):
                config_data = {}
        else:
            config_data = config_dict

        valid_fields = set(cls.model_fields.keys())
        filtered = {
            k: v
            for k, v in config_data.items()
            if k in valid_fields and v is not None
        }
        unknown = [k for k in config_data if k not in valid_fields]
        if unknown:
            logger.warning(
                f"Ignoring unknown configuration keys for {cls.__name__}: {unknown}"
            )

        return cls(**filtered)

    def cli_arg_parsers(
        self,
        choices_provider: dict[str, list[str]] | None = None,
        description: str | None = None,
        formatter_class: type[argparse.HelpFormatter] | None = None,
        required_params: set[str] | None = None,
    ) -> argparse.ArgumentParser:
        """Create an ArgumentParser populated with all config fields.

        Args:
            choices_provider: Dictionary mapping field names to valid choices.
            description: Description for the argument parser.
            formatter_class: Formatter class for the argument parser.
            required_params: Set of field names that should be required.

        Returns:
            A configured ArgumentParser.
        """
        extra_kwargs: dict[str, Any] = {}
        if formatter_class is not None:
            extra_kwargs["formatter_class"] = formatter_class

        parser = argparse.ArgumentParser(
            description=description, **extra_kwargs
        )
        choices_provider = choices_provider or {}
        required_params = (
            required_params
            if required_params is not None
            else self.get_default_required_fields()
        )

        try:
            type_hints = get_type_hints(self.__class__)
        except (NameError, AttributeError):
            type_hints = {}

        _internal_fields = {"config_file", "section_name"}
        groups: dict[str, list[tuple[str, Any]]] = {}
        ungrouped: list[tuple[str, Any]] = []

        for name, field_info in self.model_fields.items():
            if name.startswith("_") or name in _internal_fields:
                continue
            extra = field_info.json_schema_extra
            raw_group = extra.get("group") if isinstance(extra, dict) else None
            group_name = raw_group if isinstance(raw_group, str) else None
            if group_name:
                groups.setdefault(group_name, []).append((name, field_info))
            else:
                ungrouped.append((name, field_info))

        for group_name, group_fields in groups.items():
            group_desc = None
            for _, fi in group_fields:
                ex = fi.json_schema_extra
                if isinstance(ex, dict) and "group_description" in ex:
                    group_desc = ex["group_description"]
                    break
            group = parser.add_argument_group(group_name, group_desc)
            for name, fi in group_fields:
                self._add_field_as_cli_argument(
                    group,
                    name,
                    fi,
                    type_hints,
                    choices_provider,
                    required_params,
                )

        for name, fi in ungrouped:
            self._add_field_as_cli_argument(
                parser, name, fi, type_hints, choices_provider, required_params
            )

        return parser

    def _add_field_as_cli_argument(
        self,
        parser_or_group: argparse.ArgumentParser | argparse._ArgumentGroup,
        name: str,
        field_info: Any,
        type_hints: dict[str, Any],
        choices_provider: dict[str, list[str]],
        required_params: set[str],
    ) -> None:
        """Add a single pydantic field as an argparse argument."""
        field_name = name.replace("_", "-")
        arg_name = f"--{field_name}"

        field_type = type_hints.get(name, field_info.annotation)
        arg_type, action = _resolve_argparse_type(field_type)

        field_value = getattr(self, name)
        arg_kwargs: dict[str, Any] = {"default": field_value}

        if name in choices_provider:
            arg_kwargs["choices"] = choices_provider[name]

        if field_info.description:
            arg_kwargs["help"] = field_info.description

        if name in required_params:
            arg_kwargs["required"] = True

        if action:
            arg_kwargs["action"] = action
            parser_or_group.add_argument(arg_name, **arg_kwargs)
        elif get_origin(field_type) is list:
            arg_kwargs.update({"type": arg_type, "nargs": "*"})
            parser_or_group.add_argument(arg_name, **arg_kwargs)
        else:
            arg_kwargs["type"] = arg_type
            parser_or_group.add_argument(arg_name, **arg_kwargs)


class ServingBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration class for serving benchmarks (benchmark_serving.py).

    Inherits from BaseBenchmarkConfig and adds serving-specific parameters:
    - Backend and API configuration
    - Request configuration (concurrency, LoRA)
    - Traffic control (request rate, burstiness, TTFT)
    - Chat session configuration
    - Serving-specific dataset parameters
    - GPU stats collection
    """

    # Backend and API configuration (serving-specific)
    backend: Backend = Field(
        default="modular",
        description="Backend to use for benchmarking. Choices: modular, modular-chat, sglang, sglang-chat, trtllm, trtllm-chat, vllm, vllm-chat",
        json_schema_extra={
            "group": "Backend and API Configuration",
            "group_description": "Configuration for backend selection and API endpoints",
        },
    )

    base_url: str | None = Field(
        default=None,
        description="Server or API base url if not using http host and port.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    host: str = Field(
        default="localhost",
        description="Server host.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    port: int = Field(
        default=8000,
        description="Server port.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    endpoint: Endpoint = Field(
        default="/v1/chat/completions",
        description="API endpoint. Choices: /v1/completions, /v1/chat/completions, /v1/responses, /v2/models/ensemble/generate_stream",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    benchmark_task: BenchmarkTask = Field(
        default="text-generation",
        description="Benchmark task type. Choices: text-generation, text-to-image, image-to-image",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    # Request configuration (serving-specific)
    max_concurrency: str | None = Field(
        default=None,
        description="Maximum concurrent requests (optimized for serving benchmarks). Can be a single integer, 'None', or comma-separated string for sweep configs.",
        json_schema_extra={
            "group": "Request Configuration",
            "group_description": "Parameters controlling request concurrency and processing",
            "sweepable_type": "int",
        },
    )

    lora: str | None = Field(
        default=None,
        description="Optional LoRA name.",
        json_schema_extra={"group": "Request Configuration"},
    )

    # Workload configuration (serving-specific)
    max_benchmark_duration_s: int | None = Field(
        default=None,
        description="Maximum benchmark duration in seconds.",
        json_schema_extra={
            "group": "Workload Configuration",
            "group_description": "Parameters controlling benchmark duration and workload characteristics",
        },
    )

    num_chat_sessions: int | None = Field(
        default=None,
        description="Number of multiturn chat sessions.",
        json_schema_extra={"group": "Workload Configuration"},
    )

    delay_between_chat_turns: DistributionParameter | None = Field(
        default=None,
        description=(
            "Delay between chat turns in milliseconds. Accepts a float or int for a constant delay, "
            "or a distribution string: 'N(mean,std)' for normal, 'U(lower,upper)' for continuous uniform, "
            "'DU(lower,upper)' for discrete uniform, 'G(shape,scale)' for gamma, or 'LN(mean,std)' for log-normal."
        ),
        json_schema_extra={"group": "Workload Configuration"},
    )

    # Output control (serving-specific extensions)
    output_lengths: str | None = Field(
        default=None,
        description="Path to YAML file with output lengths or int.",
        json_schema_extra={
            "group": "Output Control",
            "group_description": "Parameters controlling output generation and sampling",
        },
    )

    max_output_len: int | None = Field(
        default=None,
        description="Maximum output length per request.",
        json_schema_extra={"group": "Output Control"},
    )

    temperature: float | None = Field(
        default=None,
        description="Temperature for sampling.",
        json_schema_extra={"group": "Output Control"},
    )

    top_p: float | None = Field(
        default=None,
        description="Top-p for sampling.",
        json_schema_extra={"group": "Output Control"},
    )

    top_k: int | None = Field(
        default=None,
        description="Top-k for sampling.",
        json_schema_extra={"group": "Output Control"},
    )

    # Image generation options (serving-specific)
    image_width: int | None = Field(
        default=None,
        description="Output image width in pixels for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_height: int | None = Field(
        default=None,
        description="Output image height in pixels for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_steps: int | None = Field(
        default=None,
        description="Number of denoising steps for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_guidance_scale: float | None = Field(
        default=None,
        description="Guidance scale for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_negative_prompt: str | None = Field(
        default=None,
        description="Negative prompt for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_seed: int | None = Field(
        default=None,
        description="Deterministic seed for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    # Traffic control (serving-specific)
    request_rate: str = Field(
        default="inf",
        description="Requests per second (finite rate for realistic benchmarking). Can be a single float value or comma-separated string for sweep configs.",
        json_schema_extra={
            "group": "Traffic Control",
            "group_description": "Parameters controlling request rate and traffic patterns",
            "sweepable_type": "float",
        },
    )

    burstiness: float = Field(
        default=1.0,
        description="Burstiness factor (1.0 = Poisson process).",
        json_schema_extra={"group": "Traffic Control"},
    )

    skip_first_n_requests: int | None = Field(
        default=None,
        description="Skip first N requests for measurements. Omit to auto-set to max_concurrency; pass 0 to disable.",
        json_schema_extra={"group": "Traffic Control"},
    )

    skip_last_n_requests: int | None = Field(
        default=None,
        description="Skip last N requests for measurements. Omit to auto-set to max_concurrency; pass 0 to disable.",
        json_schema_extra={"group": "Traffic Control"},
    )

    chat_warmup_delay_ms: float = Field(
        default=0.0,
        description="Delay between starting chat sessions.",
        json_schema_extra={"group": "Traffic Control"},
    )

    ignore_first_turn_stats: bool = Field(
        default=False,
        description="Ignore the first turn statistics in multiturn chat sessions.",
        json_schema_extra={"group": "Traffic Control"},
    )

    randomize_starting_turn: bool = Field(
        default=True,
        description="Start each multi-turn session at a random turn offset. Prefix turns run densely (no inter-turn delay) to build KV cache context and are excluded from benchmark results.",
        json_schema_extra={"group": "Traffic Control"},
    )

    randomize_session_start: bool = Field(
        default=True,
        description="Add a random sleep (0 to inter-turn delay) before each session's first measured query to spread out the initial wave of requests.",
        json_schema_extra={"group": "Traffic Control"},
    )

    # Dataset-specific parameters (serving workloads)
    arxiv_summarization_input_len: int = Field(
        default=15000,
        description="Number of input tokens per request, used only for arxiv-summarization dataset.",
        json_schema_extra={
            "group": "Dataset-Specific Parameters",
            "group_description": "Parameters specific to different dataset types and workloads",
        },
    )
    batch_job_image_dir: str | None = Field(
        default=None,
        description="Directory where server can access images for batch-job dataset (file reference mode). If not specified, uses embedded base64 mode.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    obfuscated_conversations_average_output_len: int = Field(
        default=175,
        description="Average output length for obfuscated-conversations dataset when output_lengths is not provided.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    obfuscated_conversations_coefficient_of_variation: float = Field(
        default=0.1,
        description="Coefficient of variation for output length for obfuscated-conversations dataset when output_lengths is not provided.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    obfuscated_conversations_shuffle: bool = Field(
        default=False,
        description="Shuffle the obfuscated-conversations dataset.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    tool_calls: bool = Field(
        default=True,
        description="Include turns with tool calls for datasets that support it. When disabled, only system+user turns are used.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_image_count: int = Field(
        default=0,
        description="Number of random images to generate.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_image_size: str = Field(
        default="",
        description="Size of random images to generate.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_input_len: DistributionParameter = Field(
        default=1024,
        description="Number of input tokens per request, used only for random sampling. Use ';' to separate first-turn and remaining-turn distributions for multiturn.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_max_num_unique_sys_prompt: int = Field(
        default=1,
        description="Maximum number of unique system prompts, used only for random sampling.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_num_turns: DistributionParameter = Field(
        default=1,
        description="Number of turns per session, used only for random sampling and --num-chat-sessions.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_output_len: DistributionParameter = Field(
        default=128,
        description="Number of output tokens per request, used only for random sampling. Use ';' to separate first-turn and remaining-turn distributions for multiturn.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_sys_prompt_ratio: float = Field(
        default=0.0,
        description="Ratio to determine the system prompt length, used only for random sampling.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    fit_distributions: bool = Field(
        default=False,
        description=(
            "With --num-chat-sessions on instruct-coder or agentic-code, reshape "
            "workloads to match random_* and delay_between_chat_turns (same "
            "semantics as random multiturn). Unsupported for code-debug multiturn. "
            "Random/synthetic datasets already follow these distributions."
        ),
        json_schema_extra={"group": "Workload Configuration"},
    )
    sonnet_input_len: int = Field(
        default=550,
        description="Number of input tokens per request, used only for sonnet dataset.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    sonnet_prefix_len: int = Field(
        default=200,
        description="Number of prefix tokens per request, used only for sonnet dataset.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )

    # Control flags (serving-specific)
    skip_test_prompt: bool = Field(
        default=False,
        description="Skip the test prompt. Useful when doing external profiling.",
        json_schema_extra={
            "group": "Control Flags",
            "group_description": "Boolean flags controlling benchmark behavior",
        },
    )
    collect_gpu_stats: bool = Field(
        default=False,
        description="Enable GPU stats collection for serving benchmarks.",
        json_schema_extra={"group": "Control Flags"},
    )

    collect_cpu_stats: bool = Field(
        default=True,
        description="Enable CPU stats collection for serving benchmarks.",
        json_schema_extra={"group": "Control Flags"},
    )

    collect_server_stats: bool = Field(
        default=True,
        description="Enable server stats collection for serving benchmarks.",
        json_schema_extra={"group": "Control Flags"},
    )

    print_workload_stats: bool = Field(
        default=False,
        description="Print workload distribution statistics (input/output lengths, num turns, delays).",
        json_schema_extra={"group": "Control Flags"},
    )

    trace: bool = Field(
        default=False,
        description="Enable nsys tracing. Requires server run under 'nsys launch'. Using '--gpu-profiling detailed' is recommended. Currently only supported on NVIDIA GPUs.",
        json_schema_extra={"group": "Control Flags"},
    )

    trace_file: str | None = Field(
        default=None,
        description="Path to save nsys trace. Default: $MODULAR_PATH/profile.nsys-rep or ./profile.nsys-rep.",
        json_schema_extra={"group": "Control Flags"},
    )

    trace_session: str | None = Field(
        default=None,
        description="Optional session name to trace. If not specified, nsys traces the default session.",
        json_schema_extra={"group": "Control Flags"},
    )

    # Result saving (serving-specific extensions)
    record_output_lengths: str | None = Field(
        default=None,
        description="Path to save output lengths in YAML format.",
        json_schema_extra={"group": "Result Saving"},
    )

    result_filename: str | None = Field(
        default=None,
        description="JSON filename for results. If None, no results are saved. Can include directory path.",
        json_schema_extra={"group": "Result Saving"},
    )

    metadata: list[str] = Field(
        default_factory=list,
        description='Key-value pairs for metadata (format: ["key=value", ...]).',
        json_schema_extra={"group": "Result Saving"},
    )

    lora_paths: list[str] = Field(
        default_factory=list,
        description="Paths to existing LoRA adapters. Format: 'path' or 'name=path'.",
        json_schema_extra={"group": "LoRA Configuration"},
    )

    lora_uniform_traffic_ratio: float = Field(
        default=0.0,
        description=(
            "Probability of selecting any LoRA uniformly at random (vs base model). "
            "Only used when per_lora_traffic_ratio is not specified. Range: 0.0-1.0."
        ),
        json_schema_extra={"group": "LoRA Configuration"},
    )

    per_lora_traffic_ratio: list[float] = Field(
        default_factory=list,
        description=(
            "Traffic percentages for each LoRA adapter in the benchmark. "
            "Must have same length as lora_paths. Sum must not exceed 1.0. "
            "Remainder goes to base model requests. "
            "If specified, overrides lora_uniform_traffic_ratio."
        ),
        json_schema_extra={"group": "LoRA Configuration"},
    )

    max_concurrent_lora_ops: int = Field(
        default=1,
        description="Maximum concurrent LoRA loading/unloading operations.",
        json_schema_extra={"group": "LoRA Configuration"},
    )

    @classmethod
    def get_default_required_fields(cls) -> set[str]:
        """Get required fields for the benchmark config."""
        return super().get_default_required_fields().union({"dataset_name"})


class SweepServingBenchmarkConfig(ServingBenchmarkConfig):
    """Configuration class for sweep serving benchmarks (sweep-benchmark-serving.py).

    Inherits from ServingBenchmarkConfig and adds sweep-specific parameters:
    - Workload configuration
    - Logging and debugging options
    - Result upload configuration
    - Sweep-specific concurrency and duration parameters
    - Metadata and result tracking
    """

    # Workload configuration (sweep-specific)
    workload_config: str = Field(
        default="",
        description="YAML file specifying the workload to benchmark.",
        json_schema_extra={
            "group": "Workload Configuration",
            "group_description": "Parameters controlling workload and dataset configuration",
        },
    )

    # Logging and debugging (sweep-specific)
    log_dir: str | None = Field(
        default=None,
        description="Path to save logs (in event of command failure only). Default: <backend>-latency-Y.m.d-H.M.S",
        json_schema_extra={
            "group": "Logging and Debugging",
            "group_description": "Parameters controlling logging and debugging behavior",
        },
    )

    dry_run: bool = Field(
        default=False,
        description="Dry run the benchmark. If true, the benchmark will not be run but all the commands that would have run will be printed.",
        json_schema_extra={"group": "Logging and Debugging"},
    )

    # Result upload configuration (sweep-specific)
    upload_results: bool = Field(
        default=False,
        description="Upload results to BigQuery.",
        json_schema_extra={
            "group": "Result Upload Configuration",
            "group_description": "Parameters controlling result upload to BigQuery",
        },
    )

    benchmark_sha: str | None = Field(
        default=None,
        description="Commit hash of the docker image used for load generation.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    cluster_information_path: str | None = Field(
        default=None,
        description="Path to the cluster information file. Usually a json file with metadata about the cluster setup if you're benchmarking more than a single node.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    benchmark_config_name: str | None = Field(
        default=None,
        description="(For serving benchmarks) config name for tracking.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    # Metadata and result tracking (sweep-specific)
    metadata: list[str] = Field(
        default_factory=list,
        description="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be saved in the result JSON file for record keeping purposes.",
        json_schema_extra={
            "group": "Metadata and Result Tracking",
            "group_description": "Parameters for metadata and result tracking",
        },
    )

    latency_percentiles: str = Field(
        default="50,90,95,99",
        description="Comma separated list of latency percentiles to include in CSV output. Only P50, P90, P95, and P99 are supported (default: 50,90,95,99).",
        json_schema_extra={"group": "Metadata and Result Tracking"},
    )

    # Sweep-specific concurrency and duration parameters
    num_iters: int = Field(
        default=1,
        description="Number of iterations to run per configuration.",
        json_schema_extra={
            "group": "Sweep Configuration",
            "group_description": "Parameters controlling sweep behavior and iteration",
        },
    )

    flush_prefix_cache: bool = Field(
        default=True,
        description="Flush the prefix cache between iterations.",
        json_schema_extra={"group": "Sweep Configuration"},
    )

    num_prompts_multiplier: int | None = Field(
        default=None,
        description=(
            "When set, num_prompts is computed as num_prompts_multiplier * max_concurrency "
            "for each concurrency level, replacing the default 300s duration timeout."
        ),
        json_schema_extra={
            "group": "Sweep Configuration",
            "cli_flag": "--num-prompts-multiplier",
        },
    )

    collect_gpu_stats: bool = Field(
        default=True,
        description="Enable GPU stats collection for serving benchmarks.",
        json_schema_extra={"group": "Control Flags"},
    )

    @classmethod
    def get_default_required_fields(cls) -> set[str]:
        """Get required fields for the sweep benchmark config."""

        # TODO: This is really lame. dataset_name is a required flag in benchmark_serving.py,
        # so you'd think it would also be required here, but it's not. This is
        # because we only parse dataset_name from the workload config file and not
        # through the command line in sweep-benchmark-serving.py. Turns out we
        # also can't quite easily pull that apart trivially when we roll this
        # part. Will circle back in a follow up PR. the --dataset-name flag
        # is set to optional here and is a no-op.
        parent_required_fields = super().get_default_required_fields()
        parent_required_fields.remove("dataset_name")
        return parent_required_fields.union({"workload_config"})


def _load_user_provided_config(
    user_config_path: Path,
    default_config_path: Path,
    config_class: type[BaseBenchmarkConfig],
) -> BaseBenchmarkConfig:
    """Load user-provided config file with inheritance from default config file.

    This function ensures that a user-provided config file inherits from a default
    config file, allowing users to override only the parameters they need
    while keeping all the default values from the base configuration.

    Args:
        user_config_path: Path to the user-provided configuration file
        default_config_path: Path to the default configuration file
        config_class: The benchmark config class to instantiate (e.g., ServingBenchmarkConfig)

    Returns:
        Config instance with inherited and overridden values
    """
    # Load the user config file
    with open(user_config_path, encoding="utf-8") as f:
        user_config_dict = yaml.safe_load(f)

    if not isinstance(user_config_dict, dict):
        raise ValueError(
            f"User configuration file {user_config_path} must contain a dictionary at the top level"
        )
    elif config_class._config_file_section_name not in user_config_dict:
        logger.warning(
            f"Cannot find {config_class._config_file_section_name} section in user configuration file {user_config_path}"
            f"Will not override benchmark config values from default config"
        )

    # Load the default config file
    with open(default_config_path, encoding="utf-8") as f:
        default_config_dict = yaml.safe_load(f)

    if not isinstance(default_config_dict, dict):
        raise ValueError(
            f"Default configuration file {default_config_path} must contain a dictionary at the top level"
        )

    # Merge the configs: user config overrides default config
    merged_config_dict = deep_merge_max_configs(
        default_config_dict, user_config_dict
    )

    # Resolve any depends_on paths relative to the default config file location
    # This is necessary because user provided configs may not have context on where
    # the "base" configs are located. This reference is only held in the default config file.
    if "depends_on" in merged_config_dict:
        depends_on_path = Path(merged_config_dict["depends_on"])
        if not depends_on_path.is_absolute():
            # Resolve relative to the default config file location
            merged_config_dict["depends_on"] = str(
                default_config_path.parent / depends_on_path
            )

    # Create a temporary config file with the merged content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        yaml.dump(merged_config_dict, temp_file)
        temp_config_path = temp_file.name

    try:
        config = config_class.from_config_file(temp_config_path)
        return config
    finally:
        # Clean up the temporary file
        Path(temp_config_path).unlink(missing_ok=True)


def parse_benchmark_args(
    config_class: type[BaseBenchmarkConfig],
    default_config_path: Path,
    description: str,
    args: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command line arguments for benchmark entrypoints with config file inheritance.

    This function first parses a preliminary argument to get the config file path,
    then loads the appropriate configuration and re-parses with the loaded config as defaults.

    Its main purpose is to handle user provided config files which override params
    of a particular benchmark entrypoint.

    Args:
        config_class: The benchmark config class to instantiate (e.g., ServingBenchmarkConfig)
        default_config_path: Path to the default configuration file. For benchmark_serving.py,
        this should be the path to the serving_config.yaml file.
        description: Description for the argument parser
        args: Command line arguments to parse. If None, parse from sys.argv.

    Returns:
        Parsed arguments namespace with config file values as defaults
    """

    # Parse the config file argument first
    config_file_path, remaining_args = (
        _resolve_user_provided_config_file_cli_arg(args=args)
    )

    if config_file_path is None:
        logger.info(
            f"No configuration file path provided, using default {default_config_path} file"
        )
        benchmark_config = config_class.from_config_file(default_config_path)
    else:
        # Check if user provided the same file as default
        if config_file_path.resolve() == default_config_path.resolve():
            logger.info(f"Using default configuration file: {config_file_path}")
            benchmark_config = config_class.from_config_file(config_file_path)
        else:
            logger.info(
                f"Using user-provided configuration file: {config_file_path} (will inherit from {default_config_path})"
            )
            # Load the user config file and ensure it inherits from default config
            benchmark_config = _load_user_provided_config(
                config_file_path, default_config_path, config_class
            )

    # Create parser using the enhanced config functionality
    # When a config file is loaded, only require parameters that are not provided in the config
    required_fields = config_class.get_default_required_fields()
    provided_required_fields = set()

    for field_name in required_fields:
        if hasattr(benchmark_config, field_name):
            field_value = getattr(benchmark_config, field_name)
            # Consider a field as "provided" if it has a non-None, non-empty value
            if field_value is not None and field_value != "":
                provided_required_fields.add(field_name)

    # Only require fields that are not provided in the config
    still_required_fields = required_fields - provided_required_fields

    parser = benchmark_config.cli_arg_parsers(
        description=description, required_params=still_required_fields
    )
    # This is added only for its help message. It's a no-op and not actually used for parsing
    # since it's done in the section above.
    parser = _add_config_file_arg_to_parser(parser)
    # Parse the remaining arguments with the loaded config as defaults
    return parser.parse_args(args=remaining_args)
