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

import subprocess
import time

import pytest
import python.runfiles
from click.testing import CliRunner
from max._entrypoints import pipelines


def test_main_help() -> None:
    """Test that the top-level help message works."""
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Commands:" in result.output


def test_help_performance() -> None:
    """This test is here to make sure that `max --help` executes quickly.

    This test has the potential to be flaky, since the time it takes to execute
    the command is dependent on the system.

    If you're here debugging this test, it's up to you to figure out if someone
    recently introduced a regression or if we simply bump the threshold.

    Regression has been because we added an import in
    pipelines.py. Importing _anything_ from MAX will cause a significant slowdown,
    so make sure to check that all imports are function local first.

    """
    THRESHOLD_MILLISECONDS = 1000

    runfiles = python.runfiles.Create()
    assert runfiles is not None, "Unable to find runfiles tree"
    loc = runfiles.Rlocation("_main/max/python/max/_entrypoints/pipelines")
    assert loc is not None, "Unable to find pipelines entrypoint"

    start_time = time.time()
    result = subprocess.run([loc, "--help"])
    assert result.returncode == 0, f"Failed to execute `{loc} --help`"

    seconds_to_milliseconds = 1000
    execution_time = (time.time() - start_time) * seconds_to_milliseconds

    print(f"`{loc} --help` executed in {execution_time:.1f} milliseconds")

    assert execution_time < THRESHOLD_MILLISECONDS, (
        f"pipelines --help command took {execution_time:.1f} milliseconds, "
        f"which exceeds the {THRESHOLD_MILLISECONDS} milliseconds threshold"
    )

    print(f"`pipelines --help` executed in {execution_time:.1f} milliseconds")


def test_serve_no_device_graph_capture_flag() -> None:
    # Regression for #83943: changing the field type of `device_graph_capture`
    # from `bool` to `bool | None` silently dropped the `--no-device-graph-capture`
    # form, breaking smoke tests and dataset eval configs that rely on it.
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--no-device-graph-capture" in result.output


def test_serve_cascade_flag_published() -> None:
    # ``max serve --cascade`` routes to the experimental Cascade server; the
    # bool must publish both flag forms so callers can opt out.
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--cascade" in result.output
    assert "--no-cascade" in result.output


def test_serve_cascade_context_options_published() -> None:
    # The Cascade deployment knobs (ContextConfig surface) must be exposed on
    # ``max serve`` so ``max serve --cascade`` can size worker pools / pick a
    # transport without falling back to the standalone cascade CLI.
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["serve", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--cascade-transport",
        "--cascade-local-cpu-workers",
        "--cascade-local-gpu-workers",
        "--cascade-remote-cpu-workers",
        "--cascade-remote-gpu-workers",
    ):
        assert flag in result.output


@pytest.mark.parametrize(
    "flag",
    [
        "--cascade-transport=http",
        "--cascade-local-cpu-workers=1",
        "--cascade-local-gpu-workers=1",
        "--cascade-remote-cpu-workers=a:9001",
        "--cascade-remote-gpu-workers=a:9001",
    ],
)
def test_serve_cascade_only_flags_require_cascade(flag: str) -> None:
    # The Cascade deployment knobs are meaningless on the standard serve path;
    # a per-option callback (with ``--cascade`` eager) rejects them before the
    # server starts instead of silently dropping them.
    runner = CliRunner()
    result = runner.invoke(
        pipelines.main, ["serve", "--model", "org/model", flag]
    )
    assert result.exit_code == 2  # Click UsageError -> exit code 2
    assert "requires --cascade" in result.output


def test_serve_cascade_only_flags_allowed_with_cascade() -> None:
    # With ``--cascade`` set, the same knobs must parse cleanly (i.e. the
    # guard callbacks do not reject them). The server itself is not started:
    # without ``--model`` the callback stops at the ``No model specified``
    # check, which runs after parsing but before the cascade branch / any
    # worker or network startup.
    runner = CliRunner()
    result = runner.invoke(
        pipelines.main,
        [
            "serve",
            "--cascade",
            "--cascade-transport",
            "grpc",
            "--cascade-local-cpu-workers",
            "1",
            "--cascade-local-gpu-workers",
            "0",
            "--cascade-remote-cpu-workers",
            "a:1",
            "--cascade-remote-gpu-workers",
            "b:1",
        ],
    )
    # Parsing passed the guard; the failure (if any) is the missing-model
    # check, never the cascade guard.
    assert "requires --cascade" not in result.output
    assert "No model specified" in result.output
