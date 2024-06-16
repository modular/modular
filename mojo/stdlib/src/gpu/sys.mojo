# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for getting NVIDIA GPUs features."""

from sys.info import _current_arch, _current_target


@always_inline("nodebug")
fn _get_sm_name() -> StringLiteral:
    return _current_arch()


@always_inline("nodebug")
fn _get_sm() -> Int:
    """Get current device SM version number."""
    alias sm_val = _get_sm_name()

    @parameter
    if sm_val == "sm_50":
        return 50
    elif sm_val == "sm_60":
        return 60
    elif sm_val == "sm_61":
        return 61
    elif sm_val == "sm_70":
        return 70
    elif sm_val == "sm_75":
        return 75
    elif sm_val == "sm_80":
        return 80
    elif sm_val == "sm_90":
        return 90
    else:
        return -1


@always_inline("nodebug")
fn is_sm[compute: IntLiteral]() -> Bool:
    """If device SM version number equals provided value."""
    return _get_sm() == compute


@always_inline("nodebug")
fn is_sm_greater[compute: Int]() -> Bool:
    """If device SM version number is greater than provided value."""
    return _get_sm() > compute


@always_inline("nodebug")
fn is_sm_greater_equal[compute: Int]() -> Bool:
    """If device SM version number is greater than or equal to the provided value.
    """
    return _get_sm() >= compute
