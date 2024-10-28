# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA compilation operations."""

import subprocess
import tempfile
from os import abort
from pathlib import Path

from compile import Info, compile_info, get_linkage_name

from .info import DEFAULT_GPU_ARCH, _get_info_from_target

# ===----------------------------------------------------------------------===#
# Targets
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_nvptx_target[
    # TODO: Ideally this is an Optional[StringLiteral] but blocked by MOCO-1039
    target_arch: StringLiteral = DEFAULT_GPU_ARCH,
]() -> __mlir_type.`!kgen.target`:
    alias info = _get_info_from_target[target_arch]()
    return info.target()


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
    compile_options: StringLiteral = "nvptx-short-ptr=true",
    is_failable: Bool = False,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
]() -> Info:
    return compile_info[
        func,
        emission_kind=emission_kind,
        compile_options=compile_options,
        is_failable=is_failable,
        target=target,
    ]()


@always_inline
fn _compile_code_asm[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    /,
    *,
    emission_kind: StringLiteral = "asm",
    compile_options: StringLiteral = "nvptx-short-ptr=true",
    is_failable: Bool = False,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
]() -> StringLiteral:
    alias asm = compile_info[
        func,
        emission_kind=emission_kind,
        compile_options=compile_options,
        is_failable=is_failable,
        target=target,
    ]().asm
    return asm


fn _get_nvptx_fn_name[
    func_type: AnyTrivialRegType, //, func: func_type
]() -> StringLiteral:
    return get_linkage_name[_get_nvptx_target(), func]()


fn _get_arch[target: __mlir_type.`!kgen.target`]() -> String:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        target,
        `, "arch" : !kgen.string`,
        `> : !kgen.string`,
    ]


@no_inline
fn _to_sass[
    target: __mlir_type.`!kgen.target` = _get_nvptx_target()
](asm: String, *, nvdisasm_opts: String = "") raises -> String:
    alias nvdisasm_path = Path("/usr/local/cuda/bin/nvdisasm")
    if not nvdisasm_path.exists():
        raise "the `nvdisasm` binary does not exist in '" + str(
            nvdisasm_path
        ) + "'"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var elf_file = Path(tmpdir) / "output.elf"
        _ = _ptxas_compile(
            asm,
            output_file=elf_file,
        )
        return subprocess.run(
            str(nvdisasm_path) + " -c " + nvdisasm_opts + " " + str(elf_file)
        )
    return ""


@no_inline
fn _ptxas_compile[
    target: __mlir_type.`!kgen.target` = _get_nvptx_target()
](
    asm: String, *, options: String = "", output_file: Optional[Path] = None
) raises -> String:
    alias ptxas_path = Path("/usr/local/cuda/bin/ptxas")
    if not ptxas_path.exists():
        raise "the `ptxas` binary does not exist in '" + str(ptxas_path) + "'"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var ptx_file = Path(tmpdir) / "output.ptx"
        var elf_file = Path(tmpdir) / "output.elf"
        ptx_file.write_text(asm)
        return subprocess.run(
            str(ptxas_path)
            + " --gpu-name "
            + _get_arch[target]()
            + " -O3 "
            + str(ptx_file)
            + " "
            + options
            + " -o "
            + str(output_file.or_else(elf_file))
        )
    return ""
