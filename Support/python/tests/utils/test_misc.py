# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import pytest

from modular.utils.misc import get_ordinal, set_env_var


def test_get_ordinal():
    assert get_ordinal(0) == "0th"
    assert get_ordinal(1) == "1st"
    assert get_ordinal(2) == "2nd"
    assert get_ordinal(3) == "3rd"
    assert get_ordinal(4) == "4th"
    assert get_ordinal(11) == "11th"
    assert get_ordinal(12) == "12th"
    assert get_ordinal(13) == "13th"
    assert get_ordinal(21) == "21st"
    assert get_ordinal(102) == "102nd"
    assert get_ordinal(1003) == "1003rd"


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


def test_set_env_var_existing(test_env_var: str):
    os.environ[test_env_var] = "somevalue"
    with set_env_var(test_env_var, "othervalue"):
        assert os.environ[test_env_var] == "othervalue"
    assert os.environ[test_env_var] == "somevalue"


def test_set_env_var_existing_unset(test_env_var: str):
    os.environ[test_env_var] = "somevalue"
    with set_env_var(test_env_var, None):
        assert os.environ.pop(test_env_var, None) is None
    assert os.environ[test_env_var] == "somevalue"


def test_set_env_var_new(test_env_var: str):
    assert os.environ.pop(test_env_var, None) is None
    with set_env_var(test_env_var, "newvalue"):
        assert os.environ[test_env_var] == "newvalue"
    assert os.environ.pop(test_env_var, None) is None


def test_set_env_var_new_unset(test_env_var: str):
    assert os.environ.pop(test_env_var, None) is None
    with set_env_var(test_env_var, None):
        assert os.environ.pop(test_env_var, None) is None
    assert os.environ.pop(test_env_var, None) is None


def test_set_env_var_remove_in_context(test_env_var: str):
    os.environ[test_env_var] = "somevalue"
    with set_env_var(test_env_var, "othervalue"):
        os.environ.pop(test_env_var)
    assert os.environ[test_env_var] == "somevalue"


def test_set_env_var_exception(test_env_var: str):
    os.environ[test_env_var] = "somevalue"

    class SomeException(Exception):
        pass

    try:
        with set_env_var(test_env_var, "othervalue"):
            raise SomeException
    except SomeException:
        pass

    assert os.environ[test_env_var] == "somevalue"
