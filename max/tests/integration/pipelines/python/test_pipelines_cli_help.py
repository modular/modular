# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import subprocess
import time

import python.runfiles
from click.testing import CliRunner
from max.entrypoints import pipelines


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
    loc = runfiles.Rlocation("_main/max/python/max/entrypoints/pipelines")
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


def test_benchmark_subcommand_help() -> None:
    """Test that the benchmark help message works."""
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["benchmark", "--help"])
    assert result.exit_code == 0

    # Check if some benchmark specific options are present.
    assert "--dataset-name" in result.output
    assert "--dataset-path" in result.output
    assert "--num-prompts" in result.output
    assert "--seed" in result.output
