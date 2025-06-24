# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements CUDA compilation operations."""

import subprocess
import tempfile
from pathlib import Path
from sys.info import _get_arch

from compile import Info, compile_info

from .info import A100, DEFAULT_GPU_ARCH
from .info import Info as HardwareInfo

# ===-----------------------------------------------------------------------===#
# Targets
# ===-----------------------------------------------------------------------===#


@always_inline
fn get_gpu_target[
    # TODO: Ideally this is an Optional[StaticString] but blocked by MOCO-1039
    target_arch: StaticString = DEFAULT_GPU_ARCH,
]() -> __mlir_type.`!kgen.target`:
    alias info = HardwareInfo.from_name[target_arch]() if target_arch else A100
    return info.target()


# ===-----------------------------------------------------------------------===#
# Compilation
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _cross_compilation() -> Bool:
    return __mlir_attr.`#kgen.param.expr<cross_compilation> : i1`


@always_inline
fn _compile_code[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    /,
    *,
    emission_kind: StaticString = "asm",
    target: __mlir_type.`!kgen.target` = get_gpu_target(),
    compile_options: StaticString = HardwareInfo.from_target[
        target
    ]().compile_options,
]() -> Info[func_type, func, target]:
    return compile_info[
        func,
        emission_kind=emission_kind,
        compile_options=compile_options,
        target=target,
    ]()


@always_inline
fn _compile_code_asm[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    /,
    *,
    emission_kind: StaticString = "asm",
    target: __mlir_type.`!kgen.target` = get_gpu_target(),
    compile_options: StaticString = HardwareInfo.from_target[
        target
    ]().compile_options,
]() -> StaticString:
    var asm = compile_info[
        func,
        emission_kind=emission_kind,
        compile_options=compile_options,
        target=target,
    ]().asm
    return asm


# ===-----------------------------------------------------------------------===#
# _to_sass
# ===-----------------------------------------------------------------------===#


@no_inline
fn _to_sass[
    target: __mlir_type.`!kgen.target` = get_gpu_target()
](asm: String, *, nvdisasm_opts: String = "") raises -> String:
    alias nvdisasm_path = Path("/usr/local/cuda/bin/nvdisasm")
    if not nvdisasm_path.exists():
        raise String(
            "the `nvdisasm` binary does not exist in '", nvdisasm_path, "'"
        )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var elf_file = Path(tmpdir) / "output.elf"
        _ = _ptxas_compile(
            asm,
            output_file=elf_file,
        )
        return subprocess.run(
            String(nvdisasm_path, " -ndf -c ", nvdisasm_opts, " ", elf_file)
        )
    return ""


# ===-----------------------------------------------------------------------===#
# _ptxas_compile
# ===-----------------------------------------------------------------------===#


@no_inline
fn _ptxas_compile[
    target: __mlir_type.`!kgen.target` = get_gpu_target()
](
    asm: String, *, options: String = "", output_file: Optional[Path] = None
) raises -> String:
    alias ptxas_path = Path("/usr/local/cuda/bin/ptxas")
    if not ptxas_path.exists():
        raise String("the `ptxas` binary does not exist in '", ptxas_path, "'")
    # Compile the PTX code to an ELF file. Here we care about the diagnostics.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var ptx_file = Path(tmpdir) / "output.ptx"
        var elf_file = Path(tmpdir) / "output.elf"
        ptx_file.write_text(asm)
        return subprocess.run(
            String(
                ptxas_path,
                " --gpu-name ",
                _get_arch[target](),
                " -O4 ",
                ptx_file,
                " ",
                options,
                " -o ",
                output_file.or_else(elf_file),
            )
        )
    return ""
