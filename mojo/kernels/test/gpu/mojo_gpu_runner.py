#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lit.formats import FileBasedTest
from math import floor
from pathlib import Path
import lit.TestRunner
import platform
import shutil

from modular.utils.subprocess import (
    run_shell_command,
    get_command_output,
    list2cmdline,
)


def has_gpu(modular_src_root, config=None):
    if config and "nvptx" not in config.available_backend_targets:
        return False

    if platform.system() != "Linux":
        return False

    try:
        get_command_output(
            [
                "mojo",
                Path(modular_src_root)
                / "Kernels"
                / "tools"
                / "cuda-query"
                / "cuda-query.mojo",
            ]
        )
        return True
    except:
        return False


def compile_gpu_kernel(
    output_path: Path, input_path: Path, compute_capability=7.5
):
    """Compiles the GPU kernel to PTX at `input_path` to the `output_path`
    containing the PTX.

    Args:
        output_path (Path): the output path.
        input_path (Path): the input path.
    """

    cmd = [
        "kgen",
        "-emit-asm",
        "--target-triple=nvptx64-nvidia-cuda",
        f"--target-cpu=sm_{floor(10*compute_capability)}",
        "--target-features=",
        str(input_path),
        "-o",
        str(output_path),
    ]
    try:
        run_shell_command(cmd)
    except Exception as e:
        print(f"Failed to run {list2cmdline(cmd)}")
        print(e)


def fix_ptx(path: Path):
    fixed_ptx = path.read_text()
    # Delete the (.param .align 1 .b8 func_retval0[0]) from the PTX
    fixed_ptx = fixed_ptx.replace(" (.param .align 1 .b8 func_retval0[0]) ", "")
    # Make the visible function an entry function
    fixed_ptx = fixed_ptx.replace(".visible .func", ".visible .entry")

    path.write_text(fixed_ptx)


class TestGPUKernel(FileBasedTest):
    """
    llvm-lit test utility for testing split-compiled kernels.
    """

    def __init__(
        self,
        config,
    ):
        self.modular_root = Path(config.modular_src_root)  # type: ignore
        self.source_root = Path(config.test_source_root)  # type: ignore
        self.exec_root = Path(config.test_exec_root)  # type: ignore

    def execute(self, test, litConfig):
        if not has_gpu(self.modular_root):
            return lit.Test.Result(
                lit.Test.EXCLUDED,
                "Test is excluded without a GPU present",
            )

        host_source_path = Path(test.getSourcePath())
        kernel_source_path = host_source_path.with_suffix(".gpu")
        if not kernel_source_path.exists():
            raise FileNotFoundError(kernel_source_path)

        # Copy the GPU file to the source directory with the .mojo extension
        copied_kernel_path = self.exec_root / (
            kernel_source_path.stem + "_gpu.mojo"
        )
        shutil.copy(kernel_source_path, copied_kernel_path)

        kernel_output_path = copied_kernel_path.with_suffix(".ptx")

        compile_gpu_kernel(kernel_output_path, copied_kernel_path)

        fix_ptx(kernel_output_path)

        extra_substitutions = [
            ("%gpu_kernel", str(kernel_output_path)),
        ]

        return lit.TestRunner.executeShTest(
            test,
            litConfig,
            False,  # execute_external
            extra_substitutions=extra_substitutions,
            preamble_commands=["mojo"],
        )
