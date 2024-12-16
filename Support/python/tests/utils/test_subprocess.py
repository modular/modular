# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import pytest
from modular.utils.subprocess import (
    CalledProcessError,
    get_command_output,
    run_shell_command,
)


def test_get_command_output():
    with pytest.raises(ValueError):
        get_command_output(["echo", __file__], capture_output=None)

    assert __file__ == get_command_output(["echo", __file__])


def test_run_shell_command_error():
    with pytest.raises(CalledProcessError):
        run_shell_command(["false"])

    run_shell_command(["false"], check=False)


def test_run_command_does_not_use_shell():
    out = get_command_output(["echo", "`not a valid shell command`"])
    assert out == "`not a valid shell command`"
