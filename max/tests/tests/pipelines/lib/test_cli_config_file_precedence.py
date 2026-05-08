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
"""Tests for Click config file precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner
from max.config import ConfigFileModel
from max.entrypoints.cli.config import config_to_flag
from pydantic import Field


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
