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

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner, Result
from smoke_tests.check_golden_eligibility import (
    IgnoredFailure,
    JobResult,
    is_ignored,
    iter_workflow_jobs,
    load_ignore_list,
    main,
    parse_model_job,
)

PREFIX = "Smoke test nightly MAX Serve container /"


# ── parse_model_job ───────────────────────────────────────────────────────────


def test_parse_b200() -> None:
    job = parse_model_job(
        f"{PREFIX} B200 - microsoft/Phi-4-mini-instruct", PREFIX
    )
    assert job is not None
    assert job.gpu == "B200"
    assert job.model == "microsoft/Phi-4-mini-instruct"
    assert job.conclusion == ""


def test_parse_mi355() -> None:
    job = parse_model_job(f"{PREFIX} MI355 - google/gemma-3-1b-it", PREFIX)
    assert job is not None
    assert job.gpu == "MI355"
    assert job.model == "google/gemma-3-1b-it"


def test_parse_8xb200() -> None:
    job = parse_model_job(
        f"{PREFIX} 8xB200 - nvidia/DeepSeek-V3.1-NVFP4", PREFIX
    )
    assert job is not None
    assert job.gpu == "8xB200"
    assert job.model == "nvidia/DeepSeek-V3.1-NVFP4"


def test_parse_8xb200_internal() -> None:
    job = parse_model_job(
        f"{PREFIX} 8xB200_internal - nvidia/Kimi-K2.5-NVFP4__internal", PREFIX
    )
    assert job is not None
    assert job.gpu == "8xB200_internal"
    assert job.model == "nvidia/Kimi-K2.5-NVFP4__internal"


def test_parse_model_with_dunder_suffix() -> None:
    job = parse_model_job(
        f"{PREFIX} MI355 - meta-llama/Llama-3.1-8B-Instruct__eagle", PREFIX
    )
    assert job is not None
    assert job.gpu == "MI355"
    assert job.model == "meta-llama/Llama-3.1-8B-Instruct__eagle"


def test_parse_commit_sha_prefix() -> None:
    # Older runs embed the commit SHA in the calling workflow's display name.
    job = parse_model_job(
        "MAX (fa3e2bd) smoke test / B200 - microsoft/phi-4",
        "MAX (fa3e2bd) smoke test /",
    )
    assert job is not None
    assert job.gpu == "B200"
    assert job.model == "microsoft/phi-4"


def test_parse_meta_job_summarize_returns_none() -> None:
    job = parse_model_job(
        f"{PREFIX} Summarize serve smoke test results", PREFIX
    )
    assert job is None


def test_parse_meta_job_decide_returns_none() -> None:
    job = parse_model_job(f"{PREFIX} Decide on models and runners", PREFIX)
    assert job is None


def test_parse_wrong_prefix_returns_none() -> None:
    job = parse_model_job(
        "Some other workflow / B200 - meta-llama/Llama-3.1-8B-Instruct", PREFIX
    )
    assert job is None


def test_parse_prefix_with_trailing_slash() -> None:
    job = parse_model_job(
        f"{PREFIX} B200 - google/gemma-3-1b-it",
        PREFIX + " /",  # extra trailing slash — should still parse
    )
    # rstrip strips the extra slash, so this should match
    assert job is not None


# ── is_ignored ────────────────────────────────────────────────────────────────


def _job(model: str, gpu: str, conclusion: str = "failure") -> JobResult:
    return JobResult(full_name="", gpu=gpu, model=model, conclusion=conclusion)


def _entry(model: str, gpu: str | None = None) -> IgnoredFailure:
    return IgnoredFailure(model=model, gpu=gpu, reason="test", ticket=None)


def test_model_not_in_list() -> None:
    assert not is_ignored(_job("org/model-A", "B200"), [_entry("org/model-B")])


def test_no_gpu_entry_matches_b200() -> None:
    assert is_ignored(_job("org/model", "B200"), [_entry("org/model")])


def test_no_gpu_entry_matches_mi355() -> None:
    assert is_ignored(_job("org/model", "MI355"), [_entry("org/model")])


def test_no_gpu_entry_does_not_match_8xb200() -> None:
    assert not is_ignored(_job("org/model", "8xB200"), [_entry("org/model")])


def test_no_gpu_entry_does_not_match_2xb200() -> None:
    assert not is_ignored(_job("org/model", "2xB200"), [_entry("org/model")])


def test_no_gpu_entry_does_not_match_4xmi355() -> None:
    assert not is_ignored(_job("org/model", "4xMI355"), [_entry("org/model")])


def test_explicit_gpu_matches_exact() -> None:
    assert is_ignored(
        _job("org/model", "8xB200"), [_entry("org/model", "8xB200")]
    )


def test_explicit_gpu_does_not_match_b200() -> None:
    assert not is_ignored(
        _job("org/model", "B200"), [_entry("org/model", "8xB200")]
    )


def test_model_name_is_case_insensitive() -> None:
    assert is_ignored(_job("ORG/Model", "B200"), [_entry("org/model")])


def test_gpu_is_case_insensitive() -> None:
    assert is_ignored(
        _job("org/model", "8xb200"), [_entry("org/model", "8xB200")]
    )


def test_empty_ignore_list() -> None:
    assert not is_ignored(_job("org/model", "B200"), [])


def test_first_matching_entry_wins() -> None:
    entries = [
        _entry("org/model", "8xB200"),  # wrong GPU for this job
        _entry("org/model"),  # matches B200
    ]
    assert is_ignored(_job("org/model", "B200"), entries)


# ── load_ignore_list ──────────────────────────────────────────────────────────


def test_load_valid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "ignore.yaml"
    p.write_text(
        textwrap.dedent("""\
            ignored_failures:
              - model: org/model-a
                reason: "known broken"
                ticket: "PROJ-123"
              - model: org/model-b
                gpu: 8xB200
                reason: "multi-gpu issue"
        """)
    )
    result = load_ignore_list(p)
    assert len(result) == 2
    assert result[0].model == "org/model-a"
    assert result[0].gpu is None
    assert result[0].ticket == "PROJ-123"
    assert result[1].model == "org/model-b"
    assert result[1].gpu == "8xB200"
    assert result[1].ticket is None


def test_load_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "ignore.yaml"
    p.write_text("ignored_failures: []\n")
    assert load_ignore_list(p) == []


def test_load_reason_defaults_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "ignore.yaml"
    p.write_text("ignored_failures:\n  - model: org/model\n")
    result = load_ignore_list(p)
    assert result[0].reason == "(no reason given)"


def test_load_missing_file_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        load_ignore_list(tmp_path / "nonexistent.yaml")
    assert exc.value.code == 1


def test_load_malformed_yaml_exits(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("ignored_failures: [: bad\n")
    with pytest.raises(SystemExit) as exc:
        load_ignore_list(p)
    assert exc.value.code == 1


# ── iter_workflow_jobs ────────────────────────────────────────────────────────


def _mock_resp(
    jobs: list[dict[str, str]],
    next_url: str | None = None,
    status: int = 200,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"jobs": jobs}
    resp.headers = {"Link": f'<{next_url}>; rel="next"' if next_url else ""}
    resp.raise_for_status = MagicMock()
    return resp


def test_iter_single_page() -> None:
    jobs = [{"name": "job1"}, {"name": "job2"}]
    with patch(
        "smoke_tests.check_golden_eligibility.requests.get",
        return_value=_mock_resp(jobs),
    ):
        assert list(iter_workflow_jobs("org/repo", "123", "tok")) == jobs


def test_iter_follows_pagination() -> None:
    page1 = [{"name": "job1"}]
    page2 = [{"name": "job2"}]
    with patch(
        "smoke_tests.check_golden_eligibility.requests.get",
        side_effect=[
            _mock_resp(page1, next_url="https://api.github.com/page2"),
            _mock_resp(page2),
        ],
    ):
        assert (
            list(iter_workflow_jobs("org/repo", "123", "tok")) == page1 + page2
        )


def test_iter_404_exits() -> None:
    with patch(
        "smoke_tests.check_golden_eligibility.requests.get",
        return_value=_mock_resp([], status=404),
    ):
        with pytest.raises(SystemExit) as exc:
            list(iter_workflow_jobs("org/repo", "bad-id", "tok"))
    assert exc.value.code == 1


# ── main (CLI integration) ────────────────────────────────────────────────────


def _raw(name: str, conclusion: str) -> dict[str, str]:
    return {"name": name, "conclusion": conclusion}


def _invoke(
    raw_jobs: list[dict[str, str]],
    ignore_yaml: str = "ignored_failures: []\n",
    extra_args: list[str] | None = None,
) -> Result:
    runner = CliRunner()
    with runner.isolated_filesystem():
        ignore_file = Path("ignore.yaml")
        ignore_file.write_text(ignore_yaml)
        with patch(
            "smoke_tests.check_golden_eligibility.iter_workflow_jobs",
            return_value=iter(raw_jobs),
        ):
            return runner.invoke(
                main,
                [
                    "--run-id",
                    "12345",
                    "--repo",
                    "org/repo",
                    "--job-prefix",
                    PREFIX,
                    "--ignore-list",
                    str(ignore_file),
                    "--token",
                    "fake-token",
                ]
                + (extra_args or []),
            )


def test_main_all_passed() -> None:
    jobs = [
        _raw(f"{PREFIX} B200 - google/gemma-3-1b-it", "success"),
        _raw(f"{PREFIX} MI355 - google/gemma-3-1b-it", "success"),
    ]
    result = _invoke(jobs)
    assert result.exit_code == 0
    assert "[PASS]" in result.output


def test_main_ignored_failure_is_golden() -> None:
    jobs = [_raw(f"{PREFIX} B200 - microsoft/phi-4", "failure")]
    ignore = textwrap.dedent("""\
        ignored_failures:
          - model: microsoft/phi-4
            reason: "known broken"
    """)
    result = _invoke(jobs, ignore)
    assert result.exit_code == 0
    assert "~ ignored" in result.output
    assert "[PASS]" in result.output


def test_main_blocking_failure() -> None:
    jobs = [_raw(f"{PREFIX} B200 - org/unknown-model", "failure")]
    result = _invoke(jobs)
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_main_timed_out_is_blocking() -> None:
    jobs = [_raw(f"{PREFIX} B200 - org/slow-model", "timed_out")]
    result = _invoke(jobs)
    assert result.exit_code == 1


def test_main_cancelled_is_blocking() -> None:
    jobs = [_raw(f"{PREFIX} B200 - org/model", "cancelled")]
    result = _invoke(jobs)
    assert result.exit_code == 1


def test_main_skipped_is_not_blocking() -> None:
    jobs = [_raw(f"{PREFIX} B200 - google/gemma-3-1b-it", "skipped")]
    result = _invoke(jobs)
    assert result.exit_code == 0


def test_main_no_token() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        ignore_file = Path("ignore.yaml")
        ignore_file.write_text("ignored_failures: []\n")
        result = runner.invoke(
            main,
            [
                "--run-id",
                "12345",
                "--repo",
                "org/repo",
                "--job-prefix",
                PREFIX,
                "--ignore-list",
                str(ignore_file),
                "--token",
                "",
            ],
        )
    assert result.exit_code == 1


def test_main_no_matching_jobs() -> None:
    jobs = [_raw("Some other workflow / B200 - org/model", "success")]
    result = _invoke(jobs)
    assert result.exit_code == 1


def test_main_multi_gpu_failure_not_suppressed_by_no_gpu_entry() -> None:
    """8xB200 failure must NOT be suppressed by a gpu-less ignore entry."""
    jobs = [_raw(f"{PREFIX} 8xB200 - org/model", "failure")]
    ignore = textwrap.dedent("""\
        ignored_failures:
          - model: org/model
            reason: "single-card only"
    """)
    result = _invoke(jobs, ignore)
    assert result.exit_code == 1


def test_main_explicit_gpu_entry_suppresses_correct_runner() -> None:
    """Explicit gpu entry suppresses failures on that GPU and not others."""
    jobs = [
        _raw(f"{PREFIX} 8xB200 - org/model", "failure"),
        _raw(f"{PREFIX} B200 - org/model", "success"),
    ]
    ignore = textwrap.dedent("""\
        ignored_failures:
          - model: org/model
            gpu: 8xB200
            reason: "only broken on 8xB200"
    """)
    result = _invoke(jobs, ignore)
    assert result.exit_code == 0


def test_main_ticket_shown_in_output() -> None:
    jobs = [_raw(f"{PREFIX} B200 - org/model", "failure")]
    ignore = textwrap.dedent("""\
        ignored_failures:
          - model: org/model
            reason: "known issue"
            ticket: "PROJ-999"
    """)
    result = _invoke(jobs, ignore)
    assert result.exit_code == 0
    assert "PROJ-999" in result.output
