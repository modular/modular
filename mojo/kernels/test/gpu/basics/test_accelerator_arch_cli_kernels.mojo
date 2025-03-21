# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug --target-accelerator=nvidia:80 %s | FileCheck --check-prefix=CHECK-NV80 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:90 %s | FileCheck --check-prefix=CHECK-NV90 %s

from sys.info import _accelerator_arch, _is_sm_9x

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from testing import *


fn check_sm() -> Bool:
    alias v = _is_sm_9x()
    return v


def main():
    alias accelerator_arch = _accelerator_arch()

    # CHECK-NV80: ret i1 false
    # CHECK-NV90: ret i1 true
    print(
        _compile_code_asm[
            check_sm,
            emission_kind="llvm",
            target = _get_gpu_target[_accelerator_arch()](),
        ]()
    )
