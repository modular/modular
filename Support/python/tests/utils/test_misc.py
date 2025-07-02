# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from modular.utils.misc import (
    create_dir_symlink,
    set_env_var,
)

# ===----------------------------------------------------------------------=== #
# Tests for set_env_var
# ===----------------------------------------------------------------------=== #


@pytest.fixture
def test_env_var() -> str:
    # This fixture makes sure that the env var name used for testing isn't used
    # by something else.

    var = "TESTENVVAR"
    suffix = 0
    while os.environ.pop(var, None) is not None:
        suffix += 1
        var = f"TESTENVVAR{suffix}"

    return var


def test_set_env_var_existing(test_env_var: str) -> None:
    os.environ[test_env_var] = "somevalue"
    with set_env_var(test_env_var, "othervalue"):
        assert os.environ[test_env_var] == "othervalue"
    assert os.environ[test_env_var] == "somevalue"


def test_set_env_var_existing_unset(test_env_var: str) -> None:
    os.environ[test_env_var] = "somevalue"
    with set_env_var(test_env_var, None):
        assert os.environ.pop(test_env_var, None) is None
    assert os.environ[test_env_var] == "somevalue"


def test_set_env_var_new(test_env_var: str) -> None:
    assert os.environ.pop(test_env_var, None) is None
    with set_env_var(test_env_var, "newvalue"):
        assert os.environ[test_env_var] == "newvalue"
    assert os.environ.pop(test_env_var, None) is None


def test_set_env_var_new_unset(test_env_var: str) -> None:
    assert os.environ.pop(test_env_var, None) is None
    with set_env_var(test_env_var, None):
        assert os.environ.pop(test_env_var, None) is None
    assert os.environ.pop(test_env_var, None) is None


def test_set_env_var_remove_in_context(test_env_var: str) -> None:
    os.environ[test_env_var] = "somevalue"
    with set_env_var(test_env_var, "othervalue"):
        os.environ.pop(test_env_var)
    assert os.environ[test_env_var] == "somevalue"


def test_set_env_var_exception(test_env_var: str) -> None:
    os.environ[test_env_var] = "somevalue"

    class SomeException(Exception):
        pass

    try:
        with set_env_var(test_env_var, "othervalue"):
            raise SomeException
    except SomeException:
        pass

    assert os.environ[test_env_var] == "somevalue"


def test_create_dir_symlink(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    destination_dir = tmp_path / "destination_dir"

    src_dir.mkdir(parents=True, exist_ok=True)

    src_dir_content = src_dir / "hello_world.txt"
    src_dir_content.write_text("Hello World!!!")

    create_dir_symlink(destination_dir, src_dir)

    assert destination_dir.is_symlink()

    linked_files = list(destination_dir.iterdir())

    assert len(linked_files) == 1

    linked_file = linked_files[0]

    assert linked_file.name == "hello_world.txt"

    assert linked_file.read_text() == "Hello World!!!"
