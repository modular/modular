# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA compilation operations."""

from compile import Info, compile_info, get_linkage_name

# ===----------------------------------------------------------------------===#
# Targets
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_nvptx_target[
    # TODO: Ideally this is an Optional[StringLiteral] but blocked by MOCO-1039
    target_arch: StringLiteral = "sm_80",
]() -> __mlir_type.`!kgen.target`:
    # Note: features = "+ptx81" means that the kernel should be compiled using
    # PTX version 8.1. This must be less than or equal to the installed CUDA
    # driver's maximum supported PTX version. Currently we hardcode this to
    # PTX version 8.1 which means that you need to have a CUDA driver newer than
    # 530.30.02 (driver included with CUDA 12.1 toolkit).
    # The mapping from CUDA Driver to PTX version can be found by looking at the
    # PTX ISA in the versioned docs here https://developer.nvidia.com/cuda-toolkit-archive.
    @parameter
    if target_arch == "sm_80":
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_80", `,
            `features = "+ptx85", `,
            `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128> : !kgen.target`,
        ]
    elif target_arch == "sm_86":  # A10
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_86", `,
            `features = "+ptx85", `,
            `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128> : !kgen.target`,
        ]
    elif target_arch == "sm_89":  # L4
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_89", `,
            `features = "+ptx85", `,
            `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128> : !kgen.target`,
        ]

    elif target_arch == "sm_90":  # H100
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_90", `,
            `features = "+ptx85", `,
            `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128> : !kgen.target`,
        ]
    elif (
        target_arch == "sm_90a"
    ):  # H100 with accelerated wgmma and setmaxnreg, not forwards compatible
        return __mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_90a", `,
            `features = "+ptx85", `,
            `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128> : !kgen.target`,
        ]
    else:
        constrained[False, "unsupported target arch " + target_arch]()
        return abort[__mlir_type.`!kgen.target`]()


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
