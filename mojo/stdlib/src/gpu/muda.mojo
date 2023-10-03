# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module exposes the NVPTX assembly generator."""

alias NVPTXTarget = __mlir_attr.`#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", simd_bit_width = 128> : !kgen.target`


fn _compile_nvptx_asm[
    func_type: AnyType, func: func_type->asm: StringLiteral
]():
    param_return[
        __mlir_attr[
            `#kgen.param.expr<compile_assembly,`,
            NVPTXTarget,
            `, `,
            func,
            `> : !kgen.string`,
        ]
    ]


fn compile_nvptx_asm[func_type: AnyType, func: func_type]() -> StringLiteral:
    alias asm: StringLiteral
    _compile_nvptx_asm[func_type, func -> asm]()
    return asm
