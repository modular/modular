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
import json
import textwrap
from pathlib import Path

from check_logit_verification_eligibility import (
    DEFAULT_RUNNERS,
    IgnoredFailure,
    ModelVerdict,
    is_ignored,
    main,
)
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# is_ignored unit tests
# ---------------------------------------------------------------------------


def test_is_ignored_exact_runner_match() -> None:
    entry = IgnoredFailure(
        model="org/model-a",
        runner="intel-gpu-8xb200",
        reason="known broken",
        ticket=None,
    )
    verdict = ModelVerdict(
        runner="intel-gpu-8xb200", model="org/model-a", status="error"
    )
    assert is_ignored(verdict, [entry])


def test_is_ignored_case_insensitive() -> None:
    entry = IgnoredFailure(
        model="Org/Model-A", runner="Intel-GPU-B200", reason="r", ticket=None
    )
    verdict = ModelVerdict(
        runner="intel-gpu-b200", model="org/model-a", status="invalid"
    )
    assert is_ignored(verdict, [entry])


def test_is_ignored_no_runner_defaults_to_single_card() -> None:
    entry = IgnoredFailure(
        model="org/model-b", runner=None, reason="r", ticket=None
    )
    for runner in DEFAULT_RUNNERS:
        v = ModelVerdict(runner=runner, model="org/model-b", status="error")
        assert is_ignored(v, [entry]), f"should be ignored on {runner}"


def test_is_ignored_no_runner_does_not_match_multi_gpu() -> None:
    entry = IgnoredFailure(
        model="org/model-b", runner=None, reason="r", ticket=None
    )
    for runner in [
        "intel-gpu-8xb200",
        "intel-gpu-b200-multi",
        "intel-gpu-4xmi355",
    ]:
        v = ModelVerdict(runner=runner, model="org/model-b", status="error")
        assert not is_ignored(v, [entry]), f"should NOT be ignored on {runner}"


def test_is_ignored_different_model_not_ignored() -> None:
    entry = IgnoredFailure(
        model="org/model-a", runner="intel-gpu-b200", reason="r", ticket=None
    )
    verdict = ModelVerdict(
        runner="intel-gpu-b200", model="org/model-b", status="error"
    )
    assert not is_ignored(verdict, [entry])


def test_is_ignored_wrong_runner_not_ignored() -> None:
    entry = IgnoredFailure(
        model="org/model-a", runner="intel-gpu-8xb200", reason="r", ticket=None
    )
    verdict = ModelVerdict(
        runner="intel-gpu-b200", model="org/model-a", status="error"
    )
    assert not is_ignored(verdict, [entry])


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def _write_verdicts(
    tmp_path: Path, files: dict[str, dict[str, object]]
) -> Path:
    verdicts_dir = tmp_path / "verdicts"
    verdicts_dir.mkdir()
    for fname, data in files.items():
        (verdicts_dir / fname).write_text(json.dumps(data))
    return verdicts_dir


def _write_ignore_list(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "ignore_list.yaml"
    p.write_text(content)
    return p


def test_cli_all_pass(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {
            "intel-gpu-b200.json": {"org/model-a": {"status": "ok"}},
            "intel-gpu-mi355.json": {"org/model-b": {"status": "ok"}},
        },
    )
    ignore_list = _write_ignore_list(tmp_path, "ignored_failures: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[PASS]" in result.output


def test_cli_failure_blocked(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "error"}}},
    )
    ignore_list = _write_ignore_list(tmp_path, "ignored_failures: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_cli_failure_ignored_single_card(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "error"}}},
    )
    ignore_yaml = textwrap.dedent("""\
        ignored_failures:
          - model: org/model-a
            reason: known broken
    """)
    ignore_list = _write_ignore_list(tmp_path, ignore_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ignored" in result.output


def test_cli_failure_on_multi_gpu_not_covered_by_no_runner_entry(
    tmp_path: Path,
) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-8xb200.json": {"org/model-a": {"status": "invalid"}}},
    )
    # Entry has no runner → only covers single-card; should NOT suppress 8xb200
    ignore_yaml = textwrap.dedent("""\
        ignored_failures:
          - model: org/model-a
            reason: known broken on single card
    """)
    ignore_list = _write_ignore_list(tmp_path, ignore_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_cli_failure_on_multi_gpu_covered_by_explicit_entry(
    tmp_path: Path,
) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-8xb200.json": {"org/model-a": {"status": "invalid"}}},
    )
    ignore_yaml = textwrap.dedent("""\
        ignored_failures:
          - model: org/model-a
            runner: intel-gpu-8xb200
            reason: known broken on 8xb200
    """)
    ignore_list = _write_ignore_list(tmp_path, ignore_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_flake_not_blocking(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "flake"}}},
    )
    ignore_list = _write_ignore_list(tmp_path, "ignored_failures: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_infra_not_blocking(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "infra"}}},
    )
    ignore_list = _write_ignore_list(tmp_path, "ignored_failures: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_mixed_pass_and_ignored(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {
            "intel-gpu-b200.json": {
                "org/good-model": {"status": "ok"},
                "org/broken-model": {"status": "error"},
            },
        },
    )
    ignore_yaml = textwrap.dedent("""\
        ignored_failures:
          - model: org/broken-model
            runner: intel-gpu-b200
            reason: known broken
    """)
    ignore_list = _write_ignore_list(tmp_path, ignore_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output
    assert "PASS" in result.output


def test_cli_no_json_files(tmp_path: Path) -> None:
    verdicts_dir = tmp_path / "empty"
    verdicts_dir.mkdir()
    ignore_list = _write_ignore_list(tmp_path, "ignored_failures: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--ignore-list",
            str(ignore_list),
        ],
    )
    assert result.exit_code != 0
