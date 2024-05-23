# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA compilation operations."""

from compile import Info, compile_info, get_linkage_name

# ===----------------------------------------------------------------------===#
# Compilation
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_nvptx_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ]


@always_inline
fn _compile_code[
    func_type: AnyRegType, //,
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
    func_type: AnyRegType, //, func: func_type
]() -> StringLiteral:
    return get_linkage_name[_get_nvptx_target(), func]()
