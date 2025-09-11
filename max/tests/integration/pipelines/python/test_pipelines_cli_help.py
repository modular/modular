# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time

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
    """Test that the --help command executes quickly"""
    THRESHOLD_MILLISECONDS = 500
    runner = CliRunner()

    start_time = time.time()
    _ = runner.invoke(pipelines.main, ["--help"])
    seconds_to_milliseconds = 1000
    execution_time = (time.time() - start_time) * seconds_to_milliseconds

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
