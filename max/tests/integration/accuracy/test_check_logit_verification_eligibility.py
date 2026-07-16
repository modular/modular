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
    ModelVerdict,
    RequiredModel,
    is_required,
    main,
)
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# is_required unit tests
# ---------------------------------------------------------------------------


def test_is_required_exact_runner_match() -> None:
    entry = RequiredModel(
        model="org/model-a",
        runner="intel-gpu-8xb200",
        reason="production model",
    )
    verdict = ModelVerdict(
        runner="intel-gpu-8xb200", model="org/model-a", status="error"
    )
    assert is_required(verdict, [entry])


def test_is_required_case_insensitive() -> None:
    entry = RequiredModel(
        model="Org/Model-A", runner="Intel-GPU-B200", reason="r"
    )
    verdict = ModelVerdict(
        runner="intel-gpu-b200", model="org/model-a", status="invalid"
    )
    assert is_required(verdict, [entry])


def test_is_required_no_runner_defaults_to_single_card() -> None:
    entry = RequiredModel(model="org/model-b", runner=None, reason="r")
    for runner in DEFAULT_RUNNERS:
        v = ModelVerdict(runner=runner, model="org/model-b", status="error")
        assert is_required(v, [entry]), f"should be required on {runner}"


def test_is_required_no_runner_does_not_match_multi_gpu() -> None:
    entry = RequiredModel(model="org/model-b", runner=None, reason="r")
    for runner in [
        "intel-gpu-8xb200",
        "intel-gpu-b200-multi",
        "intel-gpu-4xmi355",
    ]:
        v = ModelVerdict(runner=runner, model="org/model-b", status="error")
        assert not is_required(v, [entry]), (
            f"should NOT be required on {runner}"
        )


def test_is_required_different_model_not_required() -> None:
    entry = RequiredModel(
        model="org/model-a", runner="intel-gpu-b200", reason="r"
    )
    verdict = ModelVerdict(
        runner="intel-gpu-b200", model="org/model-b", status="error"
    )
    assert not is_required(verdict, [entry])


def test_is_required_wrong_runner_not_required() -> None:
    entry = RequiredModel(
        model="org/model-a", runner="intel-gpu-8xb200", reason="r"
    )
    verdict = ModelVerdict(
        runner="intel-gpu-b200", model="org/model-a", status="error"
    )
    assert not is_required(verdict, [entry])


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


def _write_required_list(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "required_list.yaml"
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
    required_list = _write_required_list(tmp_path, "required_for_golden: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[PASS]" in result.output


def test_cli_required_model_failure_blocks(tmp_path: Path) -> None:
    """A failure on a required model blocks golden."""
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "error"}}},
    )
    required_yaml = textwrap.dedent("""\
        required_for_golden:
          - model: org/model-a
            reason: production model
    """)
    required_list = _write_required_list(tmp_path, required_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_cli_non_required_failure_does_not_block(tmp_path: Path) -> None:
    """A failure on a model NOT in the required list does not block golden."""
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "error"}}},
    )
    required_list = _write_required_list(tmp_path, "required_for_golden: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not required" in result.output


def test_cli_failure_on_multi_gpu_not_covered_by_no_runner_entry(
    tmp_path: Path,
) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-8xb200.json": {"org/model-a": {"status": "invalid"}}},
    )
    # Entry has no runner → only covers single-card; should NOT block 8xb200
    required_yaml = textwrap.dedent("""\
        required_for_golden:
          - model: org/model-a
            reason: single-card only
    """)
    required_list = _write_required_list(tmp_path, required_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0
    assert "not required" in result.output


def test_cli_failure_on_multi_gpu_blocks_with_explicit_entry(
    tmp_path: Path,
) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-8xb200.json": {"org/model-a": {"status": "invalid"}}},
    )
    required_yaml = textwrap.dedent("""\
        required_for_golden:
          - model: org/model-a
            runner: intel-gpu-8xb200
            reason: required on 8xb200
    """)
    required_list = _write_required_list(tmp_path, required_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_cli_flake_not_blocking(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "flake"}}},
    )
    required_yaml = textwrap.dedent("""\
        required_for_golden:
          - model: org/model-a
            reason: required
    """)
    required_list = _write_required_list(tmp_path, required_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_infra_not_blocking(tmp_path: Path) -> None:
    verdicts_dir = _write_verdicts(
        tmp_path,
        {"intel-gpu-b200.json": {"org/model-a": {"status": "infra"}}},
    )
    required_yaml = textwrap.dedent("""\
        required_for_golden:
          - model: org/model-a
            reason: required
    """)
    required_list = _write_required_list(tmp_path, required_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_required_passes_non_required_fails(tmp_path: Path) -> None:
    """Required model passes, non-required model fails → golden eligible."""
    verdicts_dir = _write_verdicts(
        tmp_path,
        {
            "intel-gpu-b200.json": {
                "org/good-model": {"status": "ok"},
                "org/broken-model": {"status": "error"},
            },
        },
    )
    required_yaml = textwrap.dedent("""\
        required_for_golden:
          - model: org/good-model
            runner: intel-gpu-b200
            reason: required
    """)
    required_list = _write_required_list(tmp_path, required_yaml)
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not required" in result.output
    assert "PASS" in result.output


def test_cli_no_json_files(tmp_path: Path) -> None:
    """An empty verdicts directory is treated as eligible (soft pass).

    Logit verification may not produce results if the calling workflow was
    cancelled or skipped.  We do not want a cancelled run to permanently
    block the golden tag, so the script exits 0 with a WARN message.
    """
    verdicts_dir = tmp_path / "empty"
    verdicts_dir.mkdir()
    required_list = _write_required_list(tmp_path, "required_for_golden: []\n")
    result = CliRunner().invoke(
        main,
        [
            "--verdicts-dir",
            str(verdicts_dir),
            "--required-list",
            str(required_list),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output
    assert "PASS" in result.output
