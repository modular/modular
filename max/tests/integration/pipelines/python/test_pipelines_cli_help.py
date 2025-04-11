# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time

import pytest
from click.testing import CliRunner
from max.entrypoints import pipelines


def test_main_help():
    """Test that the top-level help message works."""
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Commands:" in result.output


@pytest.mark.skip("AITLIB-318: flaky, cant recreate the error locally.")
def test_subcommand_help():
    """Test that help for each subcommand works properly."""
    # Dynamically get all registered subcommands from the main command group
    subcommands = pipelines.main.commands.keys()

    runner = CliRunner()
    for cmd in subcommands:
        result = runner.invoke(pipelines.main, [cmd, "--help"])
        assert result.exit_code == 0, f"Help for subcommand '{cmd}' failed"
        assert "Usage:" in result.output


def test_help_performance():
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
