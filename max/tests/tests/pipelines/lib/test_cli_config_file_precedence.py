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
from max.entrypoints.cli.config import config_to_flag, pipeline_config_options
from max.pipelines.lib import MAXModelConfig, PipelineRuntimeConfig
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


def _make_pipeline_parallelism_cli() -> click.Command:
    @click.command()
    @pipeline_config_options
    def cli(**config_kwargs: Any) -> None:
        click.echo(
            f"{config_kwargs.get('ep_size')}|"
            f"{config_kwargs.get('data_parallel_degree')}"
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


def test_pipeline_config_options_preserves_none_defaults() -> None:
    """Absent EP/DP flags remain unset until config resolution."""
    result = CliRunner().invoke(_make_pipeline_parallelism_cli(), [])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "None|None"


def test_pipeline_config_options_preserves_explicit_parallelism_flags() -> None:
    """Explicit EP/DP flags are passed through as concrete values."""
    result = CliRunner().invoke(
        _make_pipeline_parallelism_cli(),
        ["--ep-size", "1", "--data-parallel-degree", "1"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "1|1"


def test_parallelism_backing_fields_preserve_public_api() -> None:
    runtime = PipelineRuntimeConfig(ep_size=8)
    model = MAXModelConfig(data_parallel_degree=4)

    assert runtime.ep_size_raw == 8
    assert runtime.ep_size == 8
    assert runtime.model_dump()["ep_size"] == 8
    assert "ep_size_raw" not in runtime.model_dump()

    runtime.ep_size = None
    assert runtime.ep_size_raw is None
    assert runtime.ep_size == 1

    assert model.data_parallel_degree_raw == 4
    assert model.data_parallel_degree == 4
    model_dump = model.model_dump(include={"data_parallel_degree"})
    assert model_dump["data_parallel_degree"] == 4
    assert "data_parallel_degree_raw" not in model_dump

    model.data_parallel_degree = None
    assert model.data_parallel_degree_raw is None
    assert model.data_parallel_degree == 1
