# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for getting NVIDIA GPUs features."""

from sys.info import _current_arch, _current_target

from gpu.host.info import Info


@always_inline("nodebug")
fn _get_sm_name() -> StringLiteral:
    return _current_arch()


@always_inline("nodebug")
fn _get_compute[name: StringLiteral]() -> Int:
    @parameter
    if name == "sm_50":
        return 50
    elif name == "sm_60":
        return 60
    elif name == "sm_61":
        return 61
    elif name == "sm_70":
        return 70
    elif name == "sm_75":
        return 75
    elif name == "sm_80":
        return 80
    elif name == "sm_86":
        return 86
    elif name == "sm_89":
        return 89
    elif name == "sm_90":
        return 90
    else:
        return -1


@always_inline("nodebug")
fn _get_sm() -> Int:
    """Get current device SM version number."""
    return _get_compute[_get_sm_name()]()


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
    return compute >= _get_sm()


@always_inline("nodebug")
fn is_sm_greater_equal[name: StringLiteral]() -> Bool:
    """If device SM version number is greater than or equal to the provided value.
    """

    alias current = Info.from_target_name[_get_sm_name()]()
    alias hw = Info.from_target_name[name]()

    return current >= hw
