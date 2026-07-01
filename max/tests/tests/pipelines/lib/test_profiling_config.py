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
"""Tests for ProfilingConfig env-var promotion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from max.pipelines.lib import ProfilingConfig
from pydantic import ValidationError


def test_profiling_enabled_defaults_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", raising=False)
    cfg = ProfilingConfig()
    assert cfg.profiling_enabled is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on"])
def test_profiling_enabled_env_truthy(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", value)
    cfg = ProfilingConfig()
    assert cfg.profiling_enabled is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_profiling_enabled_explicit_true_wins_over_env_falsy(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit opt-in is not vetoed by a falsy env var (explicit wins)."""
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", value)
    cfg = ProfilingConfig(profiling_enabled=True)
    assert cfg.profiling_enabled is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_profiling_enabled_explicit_false_wins_over_env_truthy(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit opt-out is not flipped on by a truthy env var."""
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", value)
    cfg = ProfilingConfig(profiling_enabled=False)
    assert cfg.profiling_enabled is False


def test_profiling_enabled_config_file_wins_over_env_falsy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config-file opt-in lands in model_fields_set, so it beats a falsy env.

    Exercises the "or a config file" half of the explicit-wins contract: the
    constructor-kwarg tests above prove the code path, this proves a
    file-sourced value is treated as explicit too (it flows through
    ConfigFileModel.load_config_file into the input dict, hence into
    model_fields_set).
    """
    config_file = tmp_path / "profiling.yaml"
    config_file.write_text("profiling_enabled: true\n")
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", "0")
    cfg = ProfilingConfig(config_file=str(config_file))
    assert cfg.profiling_enabled is True


def test_profiling_enabled_env_empty_preserves_explicit_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", raising=False)
    cfg = ProfilingConfig(profiling_enabled=True)
    assert cfg.profiling_enabled is True


def test_profiling_enabled_env_garbage_warns_and_preserves_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", "tru")
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        cfg = ProfilingConfig()
    assert cfg.profiling_enabled is False
    assert any(
        "MODULAR_MAX_DEBUG_PROFILING_ENABLED" in rec.message
        and "'tru'" in rec.message
        for rec in caplog.records
    )


@pytest.mark.parametrize("value", [" 1 ", "\t1\n", " true\n"])
def test_profiling_enabled_env_tolerates_surrounding_whitespace(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shells/orchestrators may pad the value; padding must not defeat it."""
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", value)
    cfg = ProfilingConfig()
    assert cfg.profiling_enabled is True


def test_profiling_fields_have_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock the public defaults for the libkineto config surface."""
    monkeypatch.delenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", raising=False)
    cfg = ProfilingConfig()
    assert cfg.profiling_output_path is None
    assert cfg.profiling_dynolog_enabled is True
    assert cfg.profiling_warmup_steps == 0
    assert cfg.profiling_active_steps == 10
    assert cfg.profiling_periodic_flush_seconds == 60


def test_profiling_output_path_roundtrips_without_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Templates are stored verbatim; expansion happens at trace-write time."""
    monkeypatch.delenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", raising=False)
    cfg = ProfilingConfig(profiling_output_path="/tmp/traces/{pid}_{rank}.json")
    assert cfg.profiling_output_path == "/tmp/traces/{pid}_{rank}.json"


@pytest.mark.parametrize(
    "build",
    [
        lambda: ProfilingConfig(profiling_warmup_steps=-1),
        lambda: ProfilingConfig(profiling_active_steps=0),
        lambda: ProfilingConfig(profiling_periodic_flush_seconds=0),
    ],
    ids=["warmup<0", "active<1", "flush<1"],
)
def test_profiling_step_bounds_are_enforced(
    build: Callable[[], ProfilingConfig],
) -> None:
    """``ge`` bounds reject nonsensical step/flush counts at validation."""
    with pytest.raises(ValidationError):
        build()
