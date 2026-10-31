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
"""Implements GPU compilation operations."""

import std.subprocess
import std.tempfile
from std.os import getenv
from std.pathlib import Path
from std.sys.info import CompilationTarget

from std.compile import CompiledFunctionInfo, compile_info

from .info import default_accelerator, get_gpu_target

# ===-----------------------------------------------------------------------===#
# Compilation
# ===-----------------------------------------------------------------------===#


@inline(.nodebug)
def _cross_compilation() -> Bool:
    return __mlir_attr.`#kgen.param.expr<cross_compilation> : i1`


@inline(.always)
def _compile_code[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    *,
    emission_kind: StaticString = "asm",
    target: CompilationTarget = default_accelerator(),
    compile_options: StaticString = target.default_compile_options(),
    link_options: StaticString = "",
]() -> CompiledFunctionInfo[func_type, func, target]:
    return compile_info[
        func,
        emission_kind=emission_kind,
        compile_options=compile_options,
        link_options=link_options,
        target=target,
    ]()


# ===-----------------------------------------------------------------------===#
# _to_sass
# ===-----------------------------------------------------------------------===#


@inline(.never)
def _to_sass[
    target: CompilationTarget = default_accelerator()
](asm: String, *, nvdisasm_opts: String = "") raises -> String:
    """Compiles PTX to a cubin with `ptxas` and disassembles it to SASS.

    The tools are `/usr/local/cuda/bin/ptxas` and
    `/usr/local/cuda/bin/nvdisasm` unless the `MODULAR_PTXAS_PATH` and
    `MODULAR_NVDISASM_PATH` environment variables name other binaries, for
    example the `ptxas` the runtime compiles kernels with and the `nvdisasm`
    of the same release. Each variable is checked on its own: a set variable
    must name an existing file, and the default path is not looked at then.

    Parameters:
        target: The GPU target to compile for.

    Args:
        asm: The PTX source.
        nvdisasm_opts: Extra command-line options for `nvdisasm`.

    Returns:
        The SASS disassembly.

    Raises:
        If the selected `nvdisasm` or `ptxas` does not exist.
    """
    # Resolve both tools here, outside the `with` below: `TemporaryDirectory`
    # swallows errors raised inside its block, which would turn a missing
    # tool into empty SASS.
    var nvdisasm_path = _nvdisasm_path()
    _ = _ptxas_path()
    with std.tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var elf_file = Path(tmpdir) / "output.elf"
        _ = _ptxas_compile(
            asm,
            output_file=elf_file,
        )
        return std.subprocess.run(
            String(t"{nvdisasm_path} -ndf -c {nvdisasm_opts} {elf_file}")
        )
    return ""


# ===-----------------------------------------------------------------------===#
# Tool paths
# ===-----------------------------------------------------------------------===#


def _tool_path(
    tool: StaticString, env_var: StaticString, default: StaticString
) raises -> Path:
    """Returns the binary to run for a CUDA toolkit tool.

    Args:
        tool: The tool's name, used in error messages.
        env_var: The environment variable that overrides the default path.
        default: The path used when `env_var` is unset or empty.

    Returns:
        The path in `env_var` if it is set and non-empty, otherwise `default`.

    Raises:
        If the selected binary does not exist. The default path is only
        checked when `env_var` is unset or empty, and a set `env_var` naming
        a missing file is an error rather than a reason to use the default.
    """
    var override = getenv(String(env_var))
    if override:
        var path = Path(override)
        if not path.exists():
            raise Error(
                "the `",
                tool,
                "` binary named by `",
                env_var,
                "` does not exist in '",
                path,
                "'",
            )
        return path
    var path = Path(String(default))
    if not path.exists():
        raise Error(
            "the `",
            tool,
            "` binary does not exist in '",
            path,
            "'; set `",
            env_var,
            "` to use another one",
        )
    return path


def _ptxas_path() raises -> Path:
    """Returns the `ptxas` binary to use for SASS dumps and `ptxas` output.

    Returns:
        The path in the `MODULAR_PTXAS_PATH` environment variable if it is set
        and non-empty, otherwise `/usr/local/cuda/bin/ptxas`.

    Raises:
        If the selected binary does not exist.
    """
    return _tool_path(
        "ptxas", "MODULAR_PTXAS_PATH", "/usr/local/cuda/bin/ptxas"
    )


def _nvdisasm_path() raises -> Path:
    """Returns the `nvdisasm` binary to use for SASS dumps.

    Returns:
        The path in the `MODULAR_NVDISASM_PATH` environment variable if it is
        set and non-empty, otherwise `/usr/local/cuda/bin/nvdisasm`.

    Raises:
        If the selected binary does not exist.
    """
    return _tool_path(
        "nvdisasm", "MODULAR_NVDISASM_PATH", "/usr/local/cuda/bin/nvdisasm"
    )


# ===-----------------------------------------------------------------------===#
# _ptxas_compile
# ===-----------------------------------------------------------------------===#


@inline(.never)
def _ptxas_compile[
    target: CompilationTarget = default_accelerator()
](
    asm: String, *, options: String = "", output_file: Optional[Path] = None
) raises -> String:
    """Compiles PTX to a cubin with `ptxas` and returns its diagnostics.

    The `ptxas` binary is selected as in `_ptxas_path`: the
    `MODULAR_PTXAS_PATH` environment variable if set, otherwise
    `/usr/local/cuda/bin/ptxas`.

    Parameters:
        target: The GPU target to compile for.

    Args:
        asm: The PTX source.
        options: Extra command-line options for `ptxas`.
        output_file: Where to write the cubin. Defaults to a temporary file.

    Returns:
        The output `ptxas` printed.

    Raises:
        If `ptxas` does not exist or fails.
    """
    var ptxas_path = _ptxas_path()
    # Compile the PTX code to an ELF file. Here we care about the diagnostics.
    with std.tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        var ptx_file = Path(tmpdir) / "output.ptx"
        var elf_file = Path(tmpdir) / "output.elf"
        ptx_file.write_text(asm)
        return std.subprocess.run(
            String(
                ptxas_path,
                " --gpu-name ",
                target._arch(),
                " -O4 ",
                ptx_file,
                " ",
                options,
                " -o ",
                output_file.or_else(elf_file),
            )
        )
    return ""
