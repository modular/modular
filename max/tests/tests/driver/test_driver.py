# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.driver Python bindings."""
import max.driver as md


def test_version():
    assert md.__version__, "expected version to have value"
