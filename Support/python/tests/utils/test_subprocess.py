# ===- test_subprocess.py -------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from subprocess import CalledProcessError

import pytest

from modular.utils.subprocess import (
    get_command_output,
    run_chained_commands,
    run_shell_command,
)


def test_get_command_output():
    with pytest.raises(ValueError):
        get_command_output(["echo", __file__], capture_output=None)

    assert __file__ == get_command_output(["echo", __file__])


def test_run_chained_commands():
    numbers = [str(j) for j in range(3)]
    proc = run_chained_commands(
        (["echo", n] for n in numbers), capture_output=True
    )
    output = proc.stdout.decode("utf-8").rstrip()
    assert output == "\n".join(numbers)


def test_run_shell_command_error():
    with pytest.raises(CalledProcessError):
        run_shell_command(["false"])

    run_shell_command(["false"], check=False)
