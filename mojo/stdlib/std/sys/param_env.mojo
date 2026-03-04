# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Deprecated: Use `sys.defines` instead.

This module provides backwards-compatible aliases for the functions that have
been moved to `sys.defines`. The `env_get_*` functions have been renamed to
`get_defined_*` to better reflect that they read compile-time defines (set
via `-D`), not runtime environment variables.
"""

from .defines import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    is_defined,
)


@deprecated("use `get_defined_bool` from `sys.defines` instead")
fn env_get_bool[name: StaticString]() -> Bool:
    """Try to get a boolean-valued define. Compilation fails if the
    name is not defined or the value is not a recognized boolean
    (`True`, `False`, `1`, `0`, `on`, `off`).

    Parameters:
        name: The name of the define.

    Returns:
        A boolean parameter value.
    """
    return get_defined_bool[name]()


@deprecated("use `get_defined_bool` from `sys.defines` instead")
fn env_get_bool[name: StaticString, default: Bool]() -> Bool:
    """Try to get a boolean-valued define. If the name is not defined,
    return a default value instead. The value must be a recognized boolean
    (`True`, `False`, `1`, `0`, `on`, `off`).

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A boolean parameter value.
    """
    return get_defined_bool[name, default]()


@deprecated("use `get_defined_int` from `sys.defines` instead")
fn env_get_int[name: StaticString]() -> Int:
    """Try to get an integer-valued define. Compilation fails if the
    name is not defined.

    Parameters:
        name: The name of the define.

    Returns:
        An integer parameter value.
    """
    return get_defined_int[name]()


@deprecated("use `get_defined_int` from `sys.defines` instead")
fn env_get_int[name: StaticString, default: Int]() -> Int:
    """Try to get an integer-valued define. If the name is not defined,
    return a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        An integer parameter value.
    """
    return get_defined_int[name, default]()


@deprecated("use `get_defined_string` from `sys.defines` instead")
fn env_get_string[name: StaticString]() -> StaticString:
    """Try to get a string-valued define. Compilation fails if the
    name is not defined.

    Parameters:
        name: The name of the define.

    Returns:
        A string parameter value.
    """
    return get_defined_string[name]()


@deprecated("use `get_defined_string` from `sys.defines` instead")
fn env_get_string[name: StaticString, default: StaticString]() -> StaticString:
    """Try to get a string-valued define. If the name is not defined,
    return a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A string parameter value.
    """
    return get_defined_string[name, default]()


@deprecated("use `get_defined_dtype` from `sys.defines` instead")
fn env_get_dtype[name: StaticString, default: DType]() -> DType:
    """Try to get a DType-valued define. If the name is not defined,
    return a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A DType parameter value.
    """
    return get_defined_dtype[name, default]()
