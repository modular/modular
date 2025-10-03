# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import bindings  # type: ignore


def test_logical_result() -> None:
    assert bindings.return_logical_result_success() == True
    assert bindings.return_logical_result_failure() == False


def test_error_or_success() -> None:
    assert bindings.return_error_or_success_success() == True
    assert bindings.return_error_or_success_failure() == False


def test_error_or() -> None:
    assert bindings.return_error_or_success() == 42
    assert bindings.return_error_or_failure() is None
