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
"""Tests for Click config file precedence and recipe path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner
from max.config import ConfigFileModel
from max.config.config_file_model import _resolve_config_file
from max.entrypoints.cli.config import config_to_flag, pipeline_config_options
from pydantic import Field
from pytest import MonkeyPatch


class _TestConfig(ConfigFileModel):
    model_path: str = Field(default="")
    device_graph_capture: bool = Field(default=False)


def _make_cli() -> click.Command:
    @click.command()
    @config_to_flag(_TestConfig)
    def cli(**config_kwargs: Any) -> None:
        config = _TestConfig(**config_kwargs)
        click.echo(f"{config.model_path}|{config.device_graph_capture}")

    return cli


def _make_pipeline_cli() -> click.Command:
    @click.command()
    @pipeline_config_options
    def cli(**config_kwargs: Any) -> None:
        click.echo(
            "|".join(
                key
                for key in ("device_specs", "draft_device_specs")
                if key in config_kwargs
            )
        )

    return cli


def test_config_file_overrides_click_defaults(tmp_path: Path) -> None:
    """Config file values win over Click defaults (case 2)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_path: test-model\ndevice_graph_capture: true\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        _make_cli(),
        ["--config-file", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "test-model|True"


def test_absent_fields_keep_pydantic_defaults(tmp_path: Path) -> None:
    """Fields absent from both CLI and config file get Pydantic defaults (case 1)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_path: from-file\n", encoding="utf-8")
    result = CliRunner().invoke(
        _make_cli(),
        ["--config-file", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    # device_graph_capture not in config file -> Pydantic default (False).
    assert result.output.strip() == "from-file|False"


def test_cli_args_override_config_file(tmp_path: Path) -> None:
    """Explicit CLI args override config file values (case 3)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_path: from-file\ndevice_graph_capture: true\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        _make_cli(),
        ["--config-file", str(config_path), "--model-path", "from-cli"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "from-cli|True"


def test_implicit_devices_do_not_override_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """Absent --devices leaves device_specs to config or model defaults."""
    monkeypatch.setattr(
        "max.entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(_make_pipeline_cli(), ["--config-file", "x"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_implicit_devices_use_default_without_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """Absent --devices lets MAXModelConfig use its Pydantic default."""
    monkeypatch.setattr(
        "max.entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(_make_pipeline_cli(), [])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_explicit_devices_still_override_config_file(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "max.entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(
        _make_pipeline_cli(),
        ["--config-file", "x", "--devices", "gpu:0"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "device_specs|draft_device_specs"


def test_explicit_devices_inherited_by_draft_devices(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "max.entrypoints.cli.config.DevicesOptionType.device_specs",
        staticmethod(lambda devices: [devices]),
    )

    result = CliRunner().invoke(_make_pipeline_cli(), ["--devices", "gpu:0"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "device_specs|draft_device_specs"


def test_resolve_config_file_passthrough_for_plain_paths() -> None:
    """Non-prefixed paths are returned unchanged."""
    assert _resolve_config_file("/tmp/my_config.yaml") == "/tmp/my_config.yaml"
    assert _resolve_config_file("relative/path.yaml") == "relative/path.yaml"


def test_resolve_config_file_resolves_builtin_recipe() -> None:
    """Paths with the max/pipelines/architectures/ prefix resolve to the package."""
    resolved = _resolve_config_file(
        "max/pipelines/architectures/llama3_modulev3/recipes/llama32_1b.yaml"
    )
    assert Path(resolved).is_file()
    assert resolved.endswith(
        "pipelines/architectures/llama3_modulev3/recipes/llama32_1b.yaml"
    )


def test_resolve_config_file_raises_for_missing_recipe() -> None:
    """Non-existent recipe under the prefix raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Built-in recipe not found"):
        _resolve_config_file(
            "max/pipelines/architectures/no_such/recipes/missing.yaml"
        )


def test_config_file_with_builtin_recipe_prefix() -> None:
    """ConfigFileModel resolves and opens a built-in recipe via the prefix path."""
    import yaml

    resolved = _resolve_config_file(
        "max/pipelines/architectures/llama3_modulev3/recipes/llama32_1b.yaml"
    )
    with open(resolved) as f:
        data = yaml.safe_load(f)
    assert data["model"]["model_path"] == "meta-llama/Llama-3.2-1B-Instruct"
