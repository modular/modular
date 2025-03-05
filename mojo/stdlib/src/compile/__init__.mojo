# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides utilities for compiling and inspecting Mojo code at runtime.

This module exposes functionality for compiling individual Mojo functions and
examining their low-level implementation details. It is particularly useful for:

- Inspecting assembly, LLVM IR, or object code output
- Getting linkage names and module information
- Examining function metadata like captures
- Writing compilation output to files
- Controlling compilation options and targets

Example:
    ```mojo
    from compile import compile_info

    fn my_func():
        print("Hello")

    # Get assembly for the function
    info = compile_info[my_func]()
    print(info.asm)
    ```
"""

from .compile import Info, _internal_compile_code, compile_info
from .reflection import get_linkage_name
