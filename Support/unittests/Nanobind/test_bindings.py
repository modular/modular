# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import bindings  # type: ignore
import pytest


def test_logical_result() -> None:
    assert bindings.return_logical_result_success()
    assert not bindings.return_logical_result_failure()


def test_error_or_success() -> None:
    assert bindings.return_error_or_success_success() is None
    with pytest.raises(RuntimeError):
        bindings.return_error_or_success_failure()


def test_error_or() -> None:
    assert bindings.return_error_or_success() == 42
    with pytest.raises(RuntimeError):
        bindings.return_error_or_failure()
