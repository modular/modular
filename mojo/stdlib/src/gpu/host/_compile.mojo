# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA compilation operations."""

from os import abort

from compile import Info, compile_info, get_linkage_name
from .info import _get_info_from_target

# ===----------------------------------------------------------------------===#
# Targets
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_nvptx_target[
    # TODO: Ideally this is an Optional[StringLiteral] but blocked by MOCO-1039
    target_arch: StringLiteral = "sm_80",
]() -> __mlir_type.`!kgen.target`:
    alias info = _get_info_from_target[target_arch]()
    return info.target


# ===----------------------------------------------------------------------===#
# Compilation
# ===----------------------------------------------------------------------===#


@always_inline
fn _compile_code[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    /,
    *,
    emission_kind: StringLiteral = "asm",
    is_failable: Bool = False,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
]() -> Info:
    return compile_info[
        func,
        emission_kind=emission_kind,
        is_failable=is_failable,
        target=target,
    ]()


fn _get_nvptx_fn_name[
    func_type: AnyTrivialRegType, //, func: func_type
]() -> StringLiteral:
    return get_linkage_name[_get_nvptx_target(), func]()
