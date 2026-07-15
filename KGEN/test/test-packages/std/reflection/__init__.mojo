# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Compile-time reflection utilities for introspecting Mojo types and functions.
"""

from .function import (
    ReflectedFn,
    reflect_fn,
    get_function_name,
    get_linkage_name,
)
from .reflect import Reflected, reflect
