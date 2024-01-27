# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for looking into the assembly of a specific
function. This is very useful for kernel engineers that do not want to look
at an entire file's assembly nor do they want to invoke the kgen tool manually.
"""
from gpu.host._compile import _compile_code
from sys.info import _current_target


fn compile_code[
    func_type: AnyRegType,
    func: func_type,
    /,
    *,
    emission_kind: StringLiteral = "asm",
]() -> StringLiteral:
    """Compiles the function passed in to the assembly instruction. This is
    useful to take a peak into the function assembly without requiring one to
    invoke kgen on a file."""
    alias info = _compile_code[
        func_type, func, target = _current_target(), emission_kind=emission_kind
    ]()
    return info.asm
