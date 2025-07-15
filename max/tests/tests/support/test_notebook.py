# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import pytest
from IPython.testing.globalipapp import start_ipython
from IPython.utils.capture import capture_output
from max.support.paths import MojoCompilationError


@pytest.fixture(scope="session")
def ipython():
    ipython = start_ipython()
    import max.support.notebook  # noqa

    yield ipython


def test_mojo_run_print(ipython) -> None:  # noqa: ANN001
    with capture_output() as captured:
        ipython.run_cell_magic(
            magic_name="mojo",
            line="",
            cell="""
def main():
    print(1)
""",
        )
    assert captured.stdout.strip() == "1"


def test_compile_error(ipython) -> None:  # noqa: ANN001
    with pytest.raises(MojoCompilationError):
        ipython.run_cell_magic(
            magic_name="mojo", line="", cell='''var i: Int = "hello"'''
        )


def test_mojo_package(ipython) -> None:  # noqa: ANN001
    ipython.run_cell_magic(
        magic_name="mojo",
        line="package -o hello.mojopkg",
        cell='''def hello() -> String: return "hello"''',
    )
    assert Path("hello.mojopkg").is_file()
