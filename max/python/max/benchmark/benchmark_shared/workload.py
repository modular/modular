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

"""Loads workload YAML files and applies them to a benchmark config."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig
from max.benchmark.benchmark_shared.utils import is_castable_to_int

logger = logging.getLogger(__name__)

_PATH_SUPPORTING_WORKLOAD_KEYS = (
    "dataset-path",
    "output-lengths",
    "extra-body",
    "agentic-tool-profiles",
    "response-format",
    "tools",
)
"""Workload keys whose value may name a file rather than hold content."""

_NATIVE_RUNNER = "native"


def load_workload_file(path: Path) -> Mapping[str, object]:
    """Loads a workload YAML file's options.

    A workload file is either flat, with options at the root, or
    selects the native runner with ``runner: native`` and holds the options
    in a ``native:`` block.

    Args:
        path: The workload YAML file.

    Returns:
        The workload options, keyed by their hyphenated names.

    Raises:
        ValueError: If the file is not a mapping or selects a runner other
            than the native one. This includes when it selects the native
            runner without a ``native:`` block, holds anything besides
            ``runner`` beside that block, or the block is not a mapping.
    """
    with open(path) as workload_file:
        document = yaml.safe_load(workload_file)
    if not isinstance(document, dict):
        raise ValueError(
            f"Workload {path} must be a dict, got {type(document).__name__}"
        )
    if "runner" not in document and _NATIVE_RUNNER not in document:
        return document
    runner = document.get("runner", _NATIVE_RUNNER)
    block = document.get(_NATIVE_RUNNER)
    extra_keys = sorted(
        str(key) for key in document if key not in ("runner", _NATIVE_RUNNER)
    )
    if runner != _NATIVE_RUNNER:
        raise ValueError(
            f"Workload {path} selects runner {runner!r}; only"
            " the native runner is supported."
        )
    if block is None:
        raise ValueError(
            f"Workload {path} selects the native runner but has no"
            " 'native:' block."
        )
    if not isinstance(block, dict):
        raise ValueError(
            f"Workload {path} must hold flat options or 'runner: native' with"
            " a 'native:' mapping and nothing else; got a"
            f" `native:` block of type {type(block).__name__}."
        )
    if extra_keys:
        raise ValueError(
            f"Workload {path} must hold flat options or 'runner: native' with"
            " a 'native:' mapping and nothing else; got a 'native:' block and"
            f" extra keys {extra_keys}."
        )
    return dict(block)


def _resolve_workload_path(value: str, base_dir: Path) -> str:
    """Resolves a relative workload path against ``base_dir``.

    Falls back to the original path when nothing exists in ``base_dir``, so
    it resolves against the process CWD instead: workloads written outside
    the repo, such as kimberlite's fitted configs, name repo files by their
    repo-root path.
    """
    path = Path(os.path.expandvars(value))
    if path.is_absolute() or not (base_dir / path).exists():
        return str(path)
    return str(base_dir / path)


def apply_workload(
    args: ServingBenchmarkConfig,
    workload: Mapping[str, object],
    *,
    base_dir: Path,
) -> None:
    """Applies a workload's options to ``args``.

    Values ``args`` already sets explicitly, as from the CLI, take precedence
    over the workload's. ``num-prompts`` and ``num-prompts-multiplier`` apply
    only while ``args`` holds ``None`` for both, and
    ``max-benchmark-duration-s`` only while it holds ``None`` for it, whether
    or not that value was set explicitly. Applying any workload also sets
    ``args.skip_test_prompt``.

    Args:
        args: The benchmark configuration to update.
        workload: The workload options, keyed by their hyphenated names.
        base_dir: The directory relative paths in ``workload`` resolve
            against.
    """
    # Unvalidated YAML values; the config's validators check them on
    # assignment.
    options: dict[str, Any] = dict(workload)
    for key in _PATH_SUPPORTING_WORKLOAD_KEYS:
        value = options.get(key)
        # A key may hold the value itself -- a mapping or list, or inline
        # JSON -- rather than a path to it.
        if (
            not isinstance(value, str)
            or value.lstrip().startswith(("{", "["))
            or is_castable_to_int(value)
        ):
            continue
        options[key] = _resolve_workload_path(value, base_dir)
    # Resolve max_concurrency: CLI > YAML.
    yaml_max_concurrency = options.pop("max-concurrency", None)
    if (
        yaml_max_concurrency is not None
        and "max_concurrency" not in args.model_fields_set
    ):
        # TODO(MXTOOLS-166): validate_assignment=True makes it so that this
        # goes through Pydantic's validator, which converts as needed.
        # Workload should itself be parsed with Pydantic so we don't need to
        # defer validation like so.
        args.max_concurrency = yaml_max_concurrency
    # --num-prompts and --num-prompts-multiplier both set the request count,
    # so either one on the CLI overrides both in the YAML.
    yaml_num_prompts = options.pop("num-prompts", None)
    yaml_multiplier = options.pop("num-prompts-multiplier", None)
    if args.num_prompts is None and args.num_prompts_multiplier is None:
        if yaml_num_prompts is not None:
            args.num_prompts = int(yaml_num_prompts)
        if yaml_multiplier is not None:
            # Assign the raw value so validation rejects a fractional
            # multiplier instead of int() truncating it.
            args.num_prompts_multiplier = yaml_multiplier
    # Resolve max_benchmark_duration_s: CLI > YAML.
    w_duration = options.pop("max-benchmark-duration-s", None)
    if w_duration is not None and args.max_benchmark_duration_s is None:
        args.max_benchmark_duration_s = int(w_duration)
    _apply_workload_to_config(args, options)
    args.skip_test_prompt = True


def _apply_workload_to_config(
    config: ServingBenchmarkConfig, workload: Mapping[str, object]
) -> None:
    """Sets workload YAML values as fields on ``config``.

    Keys are converted from kebab-case to snake_case. Path objects are
    stringified and env vars in string values are expanded.

    Fields already in ``config.model_fields_set`` (that is, explicitly
    provided by the caller, whether via CLI args or direct construction) are
    left unchanged so that CLI values always take precedence over workload
    YAML.

    Args:
        config: The benchmark configuration to update.
        workload: The workload options, keyed by their hyphenated names.
    """
    for k, v in workload.items():
        field_name = k.replace("-", "_")
        if field_name not in ServingBenchmarkConfig.model_fields:
            logger.warning(f"Ignoring unknown workload key: {k}")
            continue
        if field_name in config.model_fields_set:
            logger.info(
                f"CLI flag --{k} takes precedence over workload YAML"
                f" (CLI: {getattr(config, field_name)!r},"
                f" workload: {v!r})"
            )
            continue
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, str):
            v = os.path.expandvars(v)
        # 'request-rate' from YAML can be a bare number; stringify so the
        # config before-validator (comma-split strings) runs. 'max-concurrency'
        # is handled in apply_workload instead.
        # TODO(MXTOOLS-166): This should be handled through workload YAML being
        # a Pydantic model instead, eventually.
        if field_name == "request_rate" and isinstance(v, (int, float)):
            v = str(v)
        logger.info(f"Applying workload YAML value: --{k}={v!r}")
        setattr(config, field_name, v)
